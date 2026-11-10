use minijinja::{Environment, Error as JinjaError, State};
use std::sync::Arc;

pub struct TemplateEngine {
    env: Arc<Environment<'static>>,
}

impl TemplateEngine {
    pub fn new(template_str: &str) -> Result<Self, minijinja::Error> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.add_filter("tojson", tojson_filter);
        env.add_function("raise_exception", raise_exception_func);
        env.add_template_owned("chat_template", template_str.to_string())?;
        Ok(Self { env: Arc::new(env) })
    }

    #[inline(always)]
    pub fn render_chat(
        &self,
        json_line: &str,
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        // 【关键修复】：换用最稳妥的 serde_json，保证 minijinja 绝对能读懂
        let json_val: serde_json::Value =
            serde_json::from_str(json_line).map_err(|e| format!("JSON parsing failed: {}", e))?;

        let tmpl = self.env.get_template("chat_template").unwrap();

        // 同时暴露 `messages` 与 `conversations`，让模板可以根据数据 schema 任选其一。
        // 你的 SFT 数据形如 {"conversations": [{role, content}, ...]}，
        // 模板可以写 `{% for m in conversations %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}`。
        let messages = json_val
            .get("messages")
            .or_else(|| json_val.get("conversations"));

        let context = minijinja::context! {
            messages => messages,
            conversations => json_val.get("conversations"),
            tools => json_val.get("tools"),
            add_generation_prompt => add_generation_prompt,
        };

        tmpl.render(context)
            .map_err(|e| format!("Template rendering failed: {}", e))
    }
}

fn tojson_filter(_state: &State, value: minijinja::Value) -> Result<String, JinjaError> {
    Ok(serde_json::to_string(&value).unwrap_or_else(|e| {
        // 静默退回 "{}" 会掩盖模板数据问题（如不可序列化的自定义值），
        // 记录值类型便于排查模板渲染。
        tracing::warn!(
            "tojson filter failed to serialize value (kind: {:?}): {e}; falling back to '{{}}'",
            value.kind()
        );
        "{}".to_string()
    }))
}

fn raise_exception_func(msg: String) -> Result<String, JinjaError> {
    Err(JinjaError::new(
        minijinja::ErrorKind::SyntaxError,
        format!("Template raised an exception: {}", msg),
    ))
}
