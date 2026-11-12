//! Amazon Bedrock model provider.

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::types::{
    CachePointBlock, ContentBlock as BedrockContentBlock, ConversationRole, InferenceConfiguration,
    Message as BedrockMessage, SystemContentBlock as BedrockSystemContentBlock, Tool, ToolConfiguration,
    ToolInputSchema, ToolSpecification,
};
use aws_sdk_bedrockruntime::Client;
use aws_smithy_types::Document;

use super::{Model, ModelConfig, StreamEventStream};
use crate::types::{
    content::{ContentBlock, Message, Role, SystemContentBlock},
    errors::StrandsError,
    streaming::{
        ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockDeltaToolUse, ContentBlockStart,
        ContentBlockStartEvent, ContentBlockStartToolUse, ContentBlockStopEvent, MessageStartEvent,
        MessageStopEvent, MetadataEvent, Metrics, StopReason, StreamEvent, Usage,
    },
    tools::{ToolChoice, ToolSpec},
};

const DEFAULT_MODEL_ID: &str = "us.anthropic.claude-sonnet-4-20250514-v1:0";

/// Amazon Bedrock model provider.
#[derive(Debug, Clone)]
pub struct BedrockModel {
    config: ModelConfig,
    region: Option<String>,
    cache_config: Option<crate::models::cache_config::CacheConfig>,
}

impl BedrockModel {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            config: ModelConfig::new(model_id),
            region: None,
            cache_config: None,
        }
    }

    pub fn with_config(config: ModelConfig) -> Self {
        Self {
            config,
            region: None,
            cache_config: None,
        }
    }

    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Enables prompt caching with the given configuration.
    pub fn with_cache_config(mut self, cache_config: crate::models::cache_config::CacheConfig) -> Self {
        self.cache_config = Some(cache_config);
        self
    }

    fn format_messages(&self, messages: &[Message]) -> Vec<BedrockMessage> {
        let cache_last_message = self
            .cache_config
            .as_ref()
            .map(|config| config.messages_enabled())
            .unwrap_or(false);

        messages
            .iter()
            .enumerate()
            .map(|(index, msg)| {
                let role = match msg.role {
                    Role::User => ConversationRole::User,
                    Role::Assistant => ConversationRole::Assistant,
                };

                let mut content_blocks: Vec<BedrockContentBlock> = msg
                    .content
                    .iter()
                    .filter_map(|block| self.format_content_block(block))
                    .collect();

                if cache_last_message && index + 1 == messages.len() {
                    content_blocks.push(self.cache_point_block());
                }

                BedrockMessage::builder()
                    .role(role)
                    .set_content(Some(content_blocks))
                    .build()
                    .expect("valid message")
            })
            .collect()
    }

    /// The Bedrock cache-point block. The SDK in use does not expose cache
    /// point TTLs, so the configured `ttl_value` is advisory only.
    fn cache_point_block(&self) -> BedrockContentBlock {
        BedrockContentBlock::CachePoint(
            CachePointBlock::builder()
                .r#type(aws_sdk_bedrockruntime::types::CachePointType::Default)
                .build()
                .expect("valid cache point"),
        )
    }

    fn json_to_document(value: &serde_json::Value) -> Document {
        match value {
            serde_json::Value::Null => Document::Null,
            serde_json::Value::Bool(b) => Document::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Document::Number(aws_smithy_types::Number::NegInt(i))
                } else if let Some(f) = n.as_f64() {
                    Document::Number(aws_smithy_types::Number::Float(f))
                } else {
                    Document::Null
                }
            }
            serde_json::Value::String(s) => Document::String(s.clone()),
            serde_json::Value::Array(arr) => {
                Document::Array(arr.iter().map(Self::json_to_document).collect())
            }
            serde_json::Value::Object(obj) => {
                Document::Object(obj.iter().map(|(k, v)| (k.clone(), Self::json_to_document(v))).collect())
            }
        }
    }

    fn format_content_block(&self, block: &ContentBlock) -> Option<BedrockContentBlock> {
        if let Some(ref text) = block.text {
            return Some(BedrockContentBlock::Text(text.clone()));
        }

        if let Some(ref tool_use) = block.tool_use {
            let input_doc = Self::json_to_document(&tool_use.input);

            return Some(BedrockContentBlock::ToolUse(
                aws_sdk_bedrockruntime::types::ToolUseBlock::builder()
                    .tool_use_id(&tool_use.tool_use_id)
                    .name(&tool_use.name)
                    .input(input_doc)
                    .build()
                    .expect("valid tool use"),
            ));
        }

        if let Some(ref tool_result) = block.tool_result {
            let content: Vec<aws_sdk_bedrockruntime::types::ToolResultContentBlock> = tool_result
                .content
                .iter()
                .filter_map(|c| {
                    if let Some(ref text) = c.text {
                        Some(aws_sdk_bedrockruntime::types::ToolResultContentBlock::Text(text.clone()))
                    } else if let Some(ref json_val) = c.json {
                        Some(aws_sdk_bedrockruntime::types::ToolResultContentBlock::Text(
                            serde_json::to_string(json_val).unwrap_or_default(),
                        ))
                    } else {
                        None
                    }
                })
                .collect();

            let status = match tool_result.status {
                crate::types::tools::ToolResultStatus::Success => {
                    aws_sdk_bedrockruntime::types::ToolResultStatus::Success
                }
                crate::types::tools::ToolResultStatus::Error => {
                    aws_sdk_bedrockruntime::types::ToolResultStatus::Error
                }
            };

            return Some(BedrockContentBlock::ToolResult(
                aws_sdk_bedrockruntime::types::ToolResultBlock::builder()
                    .tool_use_id(&tool_result.tool_use_id)
                    .set_content(Some(content))
                    .status(status)
                    .build()
                    .expect("valid tool result"),
            ));
        }

        None
    }

    fn format_tool_specs(&self, tool_specs: &[ToolSpec]) -> Vec<Tool> {
        tool_specs
            .iter()
            .map(|spec| {
                let input_schema_doc = Self::json_to_document(&spec.input_schema.json);

                Tool::ToolSpec(
                    ToolSpecification::builder()
                        .name(&spec.name)
                        .description(&spec.description)
                        .input_schema(ToolInputSchema::Json(input_schema_doc))
                        .build()
                        .expect("valid tool spec"),
                )
            })
            .collect()
    }

    fn format_system_prompt(&self, system_prompt: Option<&str>) -> Option<Vec<BedrockSystemContentBlock>> {
        let cache_system = self
            .cache_config
            .as_ref()
            .map(|config| config.system_prompt_enabled())
            .unwrap_or(false);

        system_prompt.map(|s| {
            let mut blocks = vec![BedrockSystemContentBlock::Text(s.to_string())];
            if cache_system {
                blocks.push(BedrockSystemContentBlock::CachePoint(
                    CachePointBlock::builder()
                        .r#type(aws_sdk_bedrockruntime::types::CachePointType::Default)
                        .build()
                        .expect("valid cache point"),
                ));
            }
            blocks
        })
    }

    fn map_stop_reason(reason: &aws_sdk_bedrockruntime::types::StopReason) -> StopReason {
        match reason {
            aws_sdk_bedrockruntime::types::StopReason::EndTurn => StopReason::EndTurn,
            aws_sdk_bedrockruntime::types::StopReason::ToolUse => StopReason::ToolUse,
            aws_sdk_bedrockruntime::types::StopReason::MaxTokens => StopReason::MaxTokens,
            aws_sdk_bedrockruntime::types::StopReason::StopSequence => StopReason::StopSequence,
            aws_sdk_bedrockruntime::types::StopReason::ContentFiltered => StopReason::ContentFiltered,
            aws_sdk_bedrockruntime::types::StopReason::GuardrailIntervened => StopReason::GuardrailIntervened,
            _ => StopReason::EndTurn,
        }
    }
}

impl Default for BedrockModel {
    fn default() -> Self {
        Self::new(DEFAULT_MODEL_ID)
    }
}

#[async_trait]
impl Model for BedrockModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn update_config(&mut self, config: ModelConfig) {
        self.config = config;
    }

    /// Counts input tokens through the native Bedrock `CountTokens` API.
    async fn count_tokens(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
    ) -> Result<u64, StrandsError> {
        let mut config_loader = aws_config::defaults(BehaviorVersion::latest());
        if let Some(ref region) = self.region {
            config_loader = config_loader.region(aws_config::Region::new(region.clone()));
        }
        let sdk_config = config_loader.load().await;
        let client = Client::new(&sdk_config);

        let converse_request = aws_sdk_bedrockruntime::types::ConverseTokensRequest::builder()
            .set_messages(Some(self.format_messages(messages)));

        let converse_request = if let Some(system) = self.format_system_prompt(system_prompt) {
            converse_request.set_system(Some(system))
        } else {
            converse_request
        }
        .build();

        let output = client
            .count_tokens()
            .model_id(&self.config.model_id)
            .input(aws_sdk_bedrockruntime::types::CountTokensInput::Converse(converse_request))
            .send()
            .await
            .map_err(|e| StrandsError::ProviderTokenCountError {
                message: format!("bedrock count_tokens failed: {}", e),
            })?;

        Ok(output.input_tokens().max(0) as u64)
    }

    fn stream<'a>(
        &'a self,
        messages: &'a [Message],
        tool_specs: Option<&'a [ToolSpec]>,
        system_prompt: Option<&'a str>,
        _tool_choice: Option<ToolChoice>,
        _system_prompt_content: Option<&'a [SystemContentBlock]>,
    ) -> StreamEventStream<'a> {
        let model_id = self.config.model_id.clone();
        let max_tokens = self.config.max_tokens.unwrap_or(4096);
        let temperature = self.config.temperature;
        let top_p = self.config.top_p;
        let stop_sequences = self.config.stop_sequences.clone();

        let formatted_messages = self.format_messages(messages);
        let formatted_tools = tool_specs.map(|specs| self.format_tool_specs(specs));
        let formatted_system = self.format_system_prompt(system_prompt);
        let region = self.region.clone();

        Box::pin(async_stream::stream! {
            let mut config_loader = aws_config::defaults(BehaviorVersion::latest());
            if let Some(ref r) = region {
                config_loader = config_loader.region(aws_config::Region::new(r.clone()));
            }
            let sdk_config = config_loader.load().await;
            let client = Client::new(&sdk_config);

            let mut inference_config = InferenceConfiguration::builder().max_tokens(max_tokens as i32);

            if let Some(temp) = temperature {
                inference_config = inference_config.temperature(temp);
            }
            if let Some(p) = top_p {
                inference_config = inference_config.top_p(p);
            }
            if let Some(ref seqs) = stop_sequences {
                inference_config = inference_config.set_stop_sequences(Some(seqs.clone()));
            }

            let mut request = client
                .converse_stream()
                .model_id(&model_id)
                .set_messages(Some(formatted_messages))
                .inference_config(inference_config.build());

            if let Some(system) = formatted_system {
                request = request.set_system(Some(system));
            }

            if let Some(tools) = formatted_tools {
                request = request.tool_config(
                    ToolConfiguration::builder()
                        .set_tools(Some(tools))
                        .build()
                        .expect("valid tool config"),
                );
            }

            let response = match request.send().await {
                Ok(resp) => resp,
                Err(e) => {
                    let err_msg = e.to_string();
                    if err_msg.contains("ThrottlingException") || err_msg.contains("throttlingException") {
                        yield Err(StrandsError::ModelThrottled { message: err_msg });
                    } else if err_msg.contains("Input is too long") || err_msg.contains("context limit") {
                        yield Err(StrandsError::ContextWindowOverflow { message: err_msg });
                    } else {
                        yield Err(StrandsError::model_error(err_msg));
                    }
                    return;
                }
            };

            let mut stream = response.stream;
            let mut has_tool_use = false;

            while let Ok(Some(event)) = stream.recv().await {
                match event {
                    aws_sdk_bedrockruntime::types::ConverseStreamOutput::MessageStart(msg) => {
                        let _role = msg.role;
                        yield Ok(StreamEvent {
                            message_start: Some(MessageStartEvent { role: Role::Assistant }),
                            ..Default::default()
                        });
                    }

                    aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStart(start) => {
                        let content_block_index = start.content_block_index as u32;
                        let block_start = if let Some(ref s) = start.start {
                            match s {
                                aws_sdk_bedrockruntime::types::ContentBlockStart::ToolUse(tu) => {
                                    has_tool_use = true;
                                    Some(ContentBlockStart {
                                        tool_use: Some(ContentBlockStartToolUse {
                                            name: tu.name.clone(),
                                            tool_use_id: tu.tool_use_id.clone(),
                                        }),
                                    })
                                }
                                _ => None,
                            }
                        } else {
                            None
                        };

                        yield Ok(StreamEvent {
                            content_block_start: Some(ContentBlockStartEvent {
                                content_block_index: Some(content_block_index),
                                start: block_start,
                            }),
                            ..Default::default()
                        });
                    }

                    aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockDelta(delta) => {
                        if let Some(ref d) = delta.delta {
                            let block_delta = match d {
                                aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(text) => {
                                    ContentBlockDelta {
                                        text: Some(text.clone()),
                                        ..Default::default()
                                    }
                                }
                                aws_sdk_bedrockruntime::types::ContentBlockDelta::ToolUse(tu) => {
                                    ContentBlockDelta {
                                        tool_use: Some(ContentBlockDeltaToolUse {
                                            input: tu.input.clone(),
                                        }),
                                        ..Default::default()
                                    }
                                }
                                _ => ContentBlockDelta::default(),
                            };

                            yield Ok(StreamEvent {
                                content_block_delta: Some(ContentBlockDeltaEvent {
                                    content_block_index: Some(delta.content_block_index as u32),
                                    delta: Some(block_delta),
                                }),
                                ..Default::default()
                            });
                        }
                    }

                    aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockStop(stop) => {
                        yield Ok(StreamEvent {
                            content_block_stop: Some(ContentBlockStopEvent {
                                content_block_index: Some(stop.content_block_index as u32),
                            }),
                            ..Default::default()
                        });
                    }

                    aws_sdk_bedrockruntime::types::ConverseStreamOutput::MessageStop(stop) => {
                        let mut stop_reason = Self::map_stop_reason(&stop.stop_reason);

                        if has_tool_use && stop_reason == StopReason::EndTurn {
                            stop_reason = StopReason::ToolUse;
                        }

                        yield Ok(StreamEvent {
                            message_stop: Some(MessageStopEvent {
                                stop_reason: Some(stop_reason),
                                additional_model_response_fields: None,
                            }),
                            ..Default::default()
                        });
                    }

                    aws_sdk_bedrockruntime::types::ConverseStreamOutput::Metadata(meta) => {
                        let usage = meta.usage.map(|u| Usage {
                            input_tokens: u.input_tokens as u32,
                            output_tokens: u.output_tokens as u32,
                            total_tokens: (u.input_tokens + u.output_tokens) as u32,
                            cache_read_input_tokens: 0,
                            cache_write_input_tokens: 0,
                        });

                        let metrics = meta.metrics.map(|m| Metrics {
                            latency_ms: m.latency_ms as u64,
                            time_to_first_byte_ms: 0,
                        });

                        yield Ok(StreamEvent {
                            metadata: Some(MetadataEvent {
                                usage,
                                metrics,
                                trace: None,
                            }),
                            ..Default::default()
                        });
                    }

                    _ => {}
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bedrock_model_creation() {
        let model = BedrockModel::new("anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(model.config().model_id, "anthropic.claude-3-sonnet-20240229-v1:0");
    }

    #[test]
    fn test_bedrock_model_default() {
        let model = BedrockModel::default();
        assert!(model.config().model_id.contains("claude"));
    }

    #[test]
    fn test_bedrock_with_region() {
        let model = BedrockModel::default().with_region("us-east-1");
        assert_eq!(model.region, Some("us-east-1".to_string()));
    }

    #[test]
    fn test_json_to_document() {
        let json = serde_json::json!({"key": "value", "num": 42});
        let doc = BedrockModel::json_to_document(&json);
        match doc {
            Document::Object(map) => {
                assert!(map.contains_key("key"));
                assert!(map.contains_key("num"));
            }
            _ => panic!("expected object"),
        }
    }
}
