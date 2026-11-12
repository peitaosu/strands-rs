//! End-to-end integration tests over the public API surface.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use strands_agents::event_loop::ModelRetryStrategy;
use strands_agents::hooks::{HookEvent, HookOrder, HookProvider, HookRegistry};
use strands_agents::middleware::{
    ExecuteToolMiddleware, InvokeModelMiddleware, InvokeModelResult, MiddlewareRegistry,
};
use strands_agents::testing::{MockModel, MockResponse};
use strands_agents::tools::AgentTool;
use strands_agents::types::streaming::StopReason;
use strands_agents::{Agent, Limits, Snapshot, StrandsError};

struct EchoTool {
    hits: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl AgentTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "Echoes input"
    }
    fn tool_spec(&self) -> strands_agents::types::tools::ToolSpec {
        strands_agents::types::tools::ToolSpec::new("echo", "Echoes input").with_input_schema(
            serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}}),
        )
    }
    async fn invoke(
        &self,
        input: serde_json::Value,
        _context: &strands_agents::tools::ToolContext,
    ) -> std::result::Result<strands_agents::tools::ToolResult2, String> {
        self.hits.fetch_add(1, Ordering::SeqCst);
        let text = input
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("default");
        Ok(strands_agents::tools::ToolResult2::success(format!("echo: {}", text)))
    }
}

fn echo_tool(hits: Arc<AtomicUsize>) -> EchoTool {
    EchoTool { hits }
}

fn agent_with_tools(hits: Arc<AtomicUsize>, responses: Vec<MockResponse>) -> Agent {
    Agent::builder()
        .model_arc(Arc::new(MockModel::new(responses)))
        .tool(echo_tool(hits))
        .unwrap()
        .build()
        .unwrap()
}

#[tokio::test]
async fn test_full_tool_cycle() {
    let hits = Arc::new(AtomicUsize::new(0));
    let mut agent = agent_with_tools(
        hits.clone(),
        vec![
            MockResponse::tool_use("echo", "t1", serde_json::json!({"text": "hi"})),
            MockResponse::text("all done"),
        ],
    );

    let result = agent.invoke_async("start").await.unwrap();

    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert_eq!(result.text(), "all done");
    assert_eq!(hits.load(Ordering::SeqCst), 1);

    let history: Vec<_> = agent
        .messages()
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|b| b.as_tool_result())
        .collect();
    assert_eq!(history.len(), 1);
    assert!(history[0].content[0].text.as_deref().unwrap().contains("echo: hi"));
}

#[tokio::test]
async fn test_hook_observes_full_lifecycle() {
    let events: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
    let sink = events.clone();

    let hits = Arc::new(AtomicUsize::new(0));
    let mut agent = agent_with_tools(
        hits,
        vec![
            MockResponse::tool_use("echo", "t1", serde_json::json!({"text": "x"})),
            MockResponse::text("done"),
        ],
    );
    agent.hooks_mut().add_callback(move |event| {
        let name = match event {
            HookEvent::BeforeInvocation(_) => "before_invocation",
            HookEvent::BeforeModelCall(_) => "before_model",
            HookEvent::MessageAdded(_) => "message_added",
            HookEvent::BeforeTools(_) => "before_tools",
            HookEvent::AfterTools(_) => "after_tools",
            HookEvent::BeforeToolCall(_) => "before_tool",
            HookEvent::AfterToolCall(_) => "after_tool",
            HookEvent::AfterModelCall(_) => "after_model",
            HookEvent::AfterInvocation(_) => "after_invocation",
            _ => "other",
        };
        sink.lock().unwrap().push(name.to_string());
    });

    agent.invoke_async("go").await.unwrap();

    let recorded = events.lock().unwrap().clone();
    assert_eq!(recorded.first().map(|s| s.as_str()), Some("before_invocation"));
    assert_eq!(recorded.last().map(|s| s.as_str()), Some("after_invocation"));
    assert!(recorded.contains(&"before_tools".to_string()));
    assert!(recorded.contains(&"after_tools".to_string()));
    assert!(recorded.contains(&"before_tool".to_string()));
    assert!(recorded.contains(&"after_tool".to_string()));

    let model_calls = recorded.iter().filter(|e| e.as_str() == "before_model").count();
    assert_eq!(model_calls, 2);
}

#[tokio::test]
async fn test_middleware_short_circuits_model() {
    struct Fixed;

    #[async_trait::async_trait]
    impl InvokeModelMiddleware for Fixed {
        async fn override_result(
            &self,
            _context: &strands_agents::middleware::InvokeModelContext,
        ) -> Option<InvokeModelResult> {
            Some(InvokeModelResult {
                stop_reason: StopReason::EndTurn,
                message: strands_agents::types::content::Message::assistant("middleware answer"),
                usage: Default::default(),
            })
        }
    }

    let mut middleware = MiddlewareRegistry::new();
    middleware.add_invoke_model(Arc::new(Fixed));

    let model = Arc::new(MockModel::new(vec![MockResponse::text("never called")]));
    let mut agent = Agent::builder()
        .model_arc(model.clone())
        .middleware_registry(middleware)
        .build()
        .unwrap();

    let result = agent.invoke_async("hi").await.unwrap();
    assert_eq!(result.text(), "middleware answer");
    assert_eq!(model.calls(), 0);
}

#[tokio::test]
async fn test_middleware_blocks_tool() {
    struct DenyAll;

    #[async_trait::async_trait]
    impl ExecuteToolMiddleware for DenyAll {
        async fn override_result(
            &self,
            context: &strands_agents::middleware::ExecuteToolContext,
        ) -> Option<strands_agents::middleware::ExecuteToolResult> {
            Some(strands_agents::middleware::ExecuteToolResult {
                result: strands_agents::types::tools::ToolResult::error(
                    &context.tool_use.tool_use_id,
                    "denied by middleware",
                ),
            })
        }
    }

    let mut middleware = MiddlewareRegistry::new();
    middleware.add_execute_tool(Arc::new(DenyAll));

    let hits = Arc::new(AtomicUsize::new(0));
    let mut agent = Agent::builder()
        .model_arc(Arc::new(MockModel::new(vec![
            MockResponse::tool_use("echo", "t1", serde_json::json!({"text": "hi"})),
            MockResponse::text("wrapped up"),
        ])))
        .tool(echo_tool(hits.clone()))
        .unwrap()
        .middleware_registry(middleware)
        .build()
        .unwrap();

    let result = agent.invoke_async("go").await.unwrap();
    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert_eq!(hits.load(Ordering::SeqCst), 0);

    let tool_results: Vec<_> = agent
        .messages()
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|b| b.as_tool_result())
        .collect();
    assert!(tool_results[0].content[0]
        .text
        .as_deref()
        .unwrap()
        .contains("denied by middleware"));
}

#[tokio::test]
async fn test_cancel_after_current_tools_completes_cycle() {
    let hits = Arc::new(AtomicUsize::new(0));
    let mut agent = agent_with_tools(
        hits.clone(),
        vec![
            MockResponse::tool_use("echo", "t1", serde_json::json!({"text": "x"})),
            MockResponse::text("after tools"),
        ],
    );

    agent.cancel("stopping", true);

    let result = agent.invoke_async("go").await.unwrap();
    assert_eq!(result.stop_reason, StopReason::Cancelled);

    // after_current_tools: the in-flight cycle completes (tool runs and is
    // recorded), then the loop stops before the next model call.
    let cancel_calls: Vec<_> = agent
        .messages()
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|b| b.as_tool_result())
        .collect();
    assert_eq!(cancel_calls.len(), 1);
}

#[tokio::test]
async fn test_snapshot_round_trip_preserves_conversation() {
    let hits = Arc::new(AtomicUsize::new(0));
    let mut agent = agent_with_tools(hits, vec![MockResponse::text("state to save")]);
    agent.state_mut().set("counter", 42);

    let snapshot = agent.take_snapshot("agent-1").unwrap();
    let bytes = snapshot.to_json().unwrap();

    let restored_model = Arc::new(MockModel::new(vec![MockResponse::text("restored run")]));
    let mut restored = Agent::builder()
        .model_arc(restored_model)
        .agent_id("agent-1")
        .build()
        .unwrap();
    restored
        .load_snapshot(Snapshot::from_json(&bytes).unwrap())
        .unwrap();

    assert_eq!(restored.state().get::<i32>("counter"), Some(42));
    assert_eq!(restored.messages().len(), agent.messages().len());

    let result = restored.invoke_async("continue").await.unwrap();
    assert_eq!(result.text(), "restored run");
}

#[tokio::test]
async fn test_turn_limit_carries_usage() {
    let hits = Arc::new(AtomicUsize::new(0));
    let mut agent = Agent::builder()
        .model_arc(Arc::new(MockModel::new(vec![MockResponse::tool_use(
            "echo",
            "t1",
            serde_json::json!({"text": "x"}),
        )])))
        .tool(echo_tool(hits))
        .unwrap()
        .limits(Limits {
            turns: Some(2),
            ..Default::default()
        })
        .build()
        .unwrap();

    let result = agent.invoke_async("go").await.unwrap();
    assert_eq!(result.stop_reason, StopReason::LimitTurns);
    assert!(result.hit_limit());

    // Cycle accounting lives on the metrics, matching the Python
    // event_loop_metrics contract.
    assert!(result.metrics.cycle_count >= 1);
}

#[tokio::test]
async fn test_structured_output_via_tool_call() {
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Report {
        summary: String,
        score: u32,
    }

    let model = Arc::new(MockModel::new(vec![
        MockResponse::text("working on it"),
        MockResponse::tool_use(
            "Report",
            "t1",
            serde_json::json!({"summary": "all good", "score": 9}),
        ),
    ]));

    let mut agent = Agent::builder()
        .model_arc(model)
        .structured_output_model::<Report>()
        .build()
        .unwrap();

    let result = agent.invoke_async("report please").await.unwrap();
    let output = result.structured_output.expect("structured output");
    assert_eq!(output["summary"], "all good");
    assert_eq!(output["score"], 9);
}

#[tokio::test]
async fn test_retry_strategy_used_by_agent() {
    let model = Arc::new(MockModel::new(vec![
        MockResponse::throttled(),
        MockResponse::text("recovered"),
    ]));

    let throttle_events = Arc::new(AtomicUsize::new(0));
    let sink = throttle_events.clone();

    let mut hooks = HookRegistry::new();
    hooks.add_callback(move |event| {
        if matches!(event, HookEvent::EventLoopThrottle(_)) {
            sink.fetch_add(1, Ordering::SeqCst);
        }
    });

    let mut agent = Agent::builder()
        .model_arc(model.clone())
        .hooks(hooks)
        .retry_strategy(ModelRetryStrategy::new(
            4,
            std::time::Duration::from_millis(1),
            std::time::Duration::from_millis(2),
        ))
        .build()
        .unwrap();

    let result = agent.invoke_async("go").await.unwrap();
    assert_eq!(result.text(), "recovered");
    assert_eq!(model.calls(), 2);
    assert_eq!(throttle_events.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_direct_tool_call_records_history() {
    let hits = Arc::new(AtomicUsize::new(0));
    let mut agent = agent_with_tools(hits.clone(), vec![]);

    let result = agent
        .tool()
        .invoke("echo", serde_json::json!({"text": "direct"}))
        .await
        .unwrap();

    assert!(result.is_success());
    assert_eq!(hits.load(Ordering::SeqCst), 1);

    let texts: Vec<String> = agent
        .messages()
        .iter()
        .map(|m| m.text_content())
        .collect();
    assert!(texts.iter().any(|t| t.contains("agent.tool.echo direct tool call")));
}

#[tokio::test]
async fn test_hook_provider_receives_events() {
    struct CountingProvider {
        hits: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl HookProvider for CountingProvider {
        async fn on_event(&self, _event: &mut HookEvent) {
            self.hits.fetch_add(1, Ordering::SeqCst);
        }
    }

    let hits = Arc::new(AtomicUsize::new(0));
    let mut hooks = HookRegistry::new();
    hooks.add_provider_at(CountingProvider { hits: hits.clone() }, HookOrder::SDK_FIRST);

    let mut agent = Agent::builder()
        .model_arc(Arc::new(MockModel::new(vec![MockResponse::text("ok")])))
        .hooks(hooks)
        .build()
        .unwrap();

    agent.invoke_async("go").await.unwrap();

    assert!(hits.load(Ordering::SeqCst) >= 2);
}

#[tokio::test]
async fn test_cancel_signal_shared_with_tools() {
    struct SignalProbe;

    #[async_trait::async_trait]
    impl AgentTool for SignalProbe {
        fn name(&self) -> &str {
            "probe"
        }
        fn description(&self) -> &str {
            "Observes the cancel signal"
        }
        fn tool_spec(&self) -> strands_agents::types::tools::ToolSpec {
            strands_agents::types::tools::ToolSpec::new(self.name(), self.description())
        }
        async fn invoke(
            &self,
            _input: serde_json::Value,
            context: &strands_agents::tools::ToolContext,
        ) -> std::result::Result<strands_agents::tools::ToolResult2, String> {
            if context.is_cancelled() {
                Ok(strands_agents::tools::ToolResult2::error("cancelled"))
            } else {
                Ok(strands_agents::tools::ToolResult2::success("clear"))
            }
        }
    }

    let mut agent = Agent::builder()
        .model_arc(Arc::new(MockModel::new(vec![MockResponse::text("done")])))
        .tool(SignalProbe)
        .unwrap()
        .build()
        .unwrap();

    let clear = agent
        .tool()
        .invoke("probe", serde_json::json!({}))
        .await
        .unwrap();
    assert!(clear.is_success());

    agent.cancel_signal().set();

    let cancelled = agent
        .tool()
        .invoke("probe", serde_json::json!({}))
        .await
        .unwrap();
    assert!(cancelled.is_error());
}

#[test]
fn test_limits_builder_wiring() {
    let agent = Agent::builder()
        .model_arc(Arc::new(MockModel::new(vec![])))
        .limits(Limits {
            turns: Some(3),
            ..Default::default()
        })
        .build()
        .unwrap();

    assert_eq!(agent.limits().turns, Some(3));
}

#[test]
fn test_agent_result_error_shapes() {
    let error = StrandsError::MaxTokensReached {
        message: "limit hit".to_string(),
    };
    assert!(error.to_string().contains("limit hit"));

    let error = StrandsError::SandboxError {
        message: "boom".to_string(),
    };
    assert!(error.to_string().contains("Sandbox"));
}
