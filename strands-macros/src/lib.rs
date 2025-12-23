//! Procedural macros for Strands agents.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Attribute, FnArg, ItemFn, Pat};

/// Transforms a function into a Strands agent tool.
///
/// # Example
///
/// ```rust,ignore
/// use strands::tool;
///
/// #[tool]
/// /// Get the current weather for a location.
/// async fn get_weather(location: String, units: Option<String>) -> String {
///     format!("Weather in {}: 72°F", location)
/// }
/// ```
///
/// The generated tool can be used with an agent:
///
/// ```rust,ignore
/// let agent = Agent::builder()
///     .model(BedrockModel::default())
///     .tools(vec![GetWeatherTool::new()])
///     .build()?;
/// ```
#[proc_macro_attribute]
pub fn tool(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let struct_name = format_ident!("{}Tool", to_pascal_case(&fn_name_str));

    let description = extract_doc_comment(&input_fn.attrs);

    let params: Vec<_> = input_fn
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg {
                if let Pat::Ident(pat_ident) = &*pat_type.pat {
                    let name = pat_ident.ident.to_string();
                    if name == "tool_context" || name == "agent" {
                        return None;
                    }
                    let ty = &pat_type.ty;
                    return Some((name, quote!(#ty).to_string()));
                }
            }
            None
        })
        .collect();

    let param_schemas: Vec<_> = params
        .iter()
        .map(|(name, _ty)| {
            quote! {
                properties.insert(
                    #name.to_string(),
                    serde_json::json!({
                        "type": "string",
                        "description": format!("Parameter {}", #name)
                    })
                );
            }
        })
        .collect();

    let required_params: Vec<_> = params
        .iter()
        .filter(|(_, ty)| !ty.contains("Option"))
        .map(|(name, _)| name.clone())
        .collect();

    let is_async = input_fn.sig.asyncness.is_some();

    let param_names: Vec<_> = params
        .iter()
        .map(|(name, _)| format_ident!("{}", name))
        .collect();

    let execute_call = if is_async {
        quote! { #fn_name(#(#param_names),*).await }
    } else {
        quote! { #fn_name(#(#param_names),*) }
    };

    let param_extractions: Vec<_> = params
        .iter()
        .map(|(name, ty)| {
            let ident = format_ident!("{}", name);
            if ty.contains("Option") {
                quote! {
                    let #ident = tool_use.input.get(#name)
                        .and_then(|v| serde_json::from_value(v.clone()).ok());
                }
            } else {
                quote! {
                    let #ident = tool_use.input.get(#name)
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .ok_or_else(|| format!("Missing required parameter: {}", #name))?;
                }
            }
        })
        .collect();

    let expanded = quote! {
        #input_fn

        #[derive(Clone)]
        pub struct #struct_name;

        impl #struct_name {
            pub fn new() -> Self { Self }
        }

        impl Default for #struct_name {
            fn default() -> Self { Self::new() }
        }

        impl strands::tools::AgentTool for #struct_name {
            fn tool_name(&self) -> &str {
                #fn_name_str
            }

            fn tool_spec(&self) -> strands::types::tools::ToolSpec {
                let mut properties = std::collections::HashMap::new();
                #(#param_schemas)*

                let required: Vec<String> = vec![#(#required_params.to_string()),*];

                strands::types::tools::ToolSpec {
                    name: #fn_name_str.to_string(),
                    description: #description.to_string(),
                    input_schema: strands::types::tools::InputSchema {
                        json: serde_json::json!({
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }),
                    },
                    output_schema: None,
                }
            }

            fn tool_type(&self) -> &str {
                "function"
            }

            fn stream(
                &self,
                tool_use: &strands::types::tools::ToolUse,
                _invocation_state: &strands::tools::InvocationState,
            ) -> strands::tools::ToolGenerator {
                let tool_use = tool_use.clone();
                Box::pin(async_stream::stream! {
                    let result: Result<String, String> = (|| async {
                        #(#param_extractions)*
                        let output = #execute_call;
                        Ok(output.to_string())
                    })().await;

                    let tool_result = match result {
                        Ok(text) => strands::types::tools::ToolResult::success(
                            &tool_use.tool_use_id,
                            text,
                        ),
                        Err(e) => strands::types::tools::ToolResult::error(
                            &tool_use.tool_use_id,
                            e,
                        ),
                    };
                    yield strands::tools::ToolEvent::Result(tool_result);
                })
            }
        }
    };

    TokenStream::from(expanded)
}

fn extract_doc_comment(attrs: &[Attribute]) -> String {
    let mut doc_lines = Vec::new();

    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(meta) = &attr.meta {
                if let syn::Expr::Lit(expr_lit) = &meta.value {
                    if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                        doc_lines.push(lit_str.value().trim().to_string());
                    }
                }
            }
        }
    }

    let mut description = Vec::new();
    for line in doc_lines {
        let lower = line.to_lowercase();
        if lower.starts_with("# arg") || lower.starts_with("args:") || lower.starts_with("arguments:") {
            break;
        }
        description.push(line);
    }

    description.join(" ").trim().to_string()
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().chain(chars).collect(),
                None => String::new(),
            }
        })
        .collect()
}
