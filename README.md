# ds4

`ds4` is a native inference engine for DeepSeek V4 Flash. It is
intentionally narrow: not a generic GGUF runner, not a wrapper around another
runtime, and not a framework. The inference path is a DeepSeek V4 Flash-specific
Metal GPU graph executor with DS4-specific loading, prompt rendering, KV state,
and server API glue. The app layer is written in Rust; the engine core and Metal
dispatch are in C.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

Why we believe DeepSeek v4 Flash to be a pretty special model deserving a stand
alone engine? Because after comparing it with powerful smaller dense models, we
can report that:

1. DeepSeek v4 Flash is faster because of less active parameters.
2. In thinking mode, if you avoid *max thinking*, it produces a thinking section that is a lot shorter than other models, even 1/5 of other models in many cases, and crucially, the thinking section length is **proportional to the problem complexity**. This makes DeepSeek v4 Flash usable with thinking enabled when other models are practically impossible to use in the same conditions.
3. The model features a context window of **1 million tokens**.
4. Being so large, it knows more things if you go sampling at the edge of knowledge. For instance asking about Italian show or political questions soon uncovers that 284B parameters are a lot more than 27B or 35B parameters.
5. It writes much better English and Italian. It *feels* a quasi-frontier model.
6. The KV cache is incredibly compressed, allowing long context inference on local computers and **on disk KV cache persistence**.
7. It works well with 2-bit quantization, if quantized in a special way (read later). This allows to run it in MacBooks with 128GB of RAM.
8. We expect DeepSeek to release **updated versions of v4 Flash** in the future, even better than the current one.

That said, a few important things about this project:

* The local inference landscape contains many excellent projects, but new models are released continuously, and the attention immediately gets captured by the next model to implement. This project takes a deliberately narrow bet: one model at a time, official-vector validation (logits obtained with the official implementation), long-context tests, and enough agent integration to know if it really works. The exact model may change as the landscape evolves, but the constraint remains: local inference credible on high end personal machines or Mac Studios, starting from 128GB of memory.
* This software is developed with **strong assistance from GPT 5.5** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.
* This implementation is based on the idea that compressed KV caches like the one of DeepSeek v4 and the fast SSD disks of modern MacBooks should change our idea that KV cache belongs to RAM. **The KV cache is actually a first class disk citizen**.
* Our vision is that local inference should be a set of three things working well together, out of the box: A) inference engine with HTTP API + B) GGUF specially crafted to run well under a given engine and given assumptions + C) testing and validation with coding agents implementations. This inference engine only runs with the GGUF files provided. It gets tested against officially obtained logits at different context sizes. This project exists because we wanted to make one local model feel finished end to end, not just runnable. However this is just alpha quality code, so probably we are not still there.
* This is **Metal-only**, may implement CUDA support in the future? Perhaps, but nothing more. The CPU path is only for correctness check, but **warning: current macOS versions have a bug in the virtual memory implementation that will crash the kernel** if you try to run the CPU code. Remember? Software sucks.

## Acknowledgements to llama.cpp and GGML

`ds4` does not link against GGML, but it **exists thanks to the path opened by
the llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and
hard-won engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 Flash-specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain Metal kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Model Weights

This implementation only works with the DeepSeek V4 Flash GGUFs published for
this project. It is not a general GGUF loader, and arbitrary DeepSeek/GGUF files
will not have the tensor layout, quantization mix, metadata, or optional MTP
state expected by the engine. The 2 bit quantizations provided here are not
a joke: they behave well, work under coding agents, call tools in a reliable way.
The 2 bit quants use a very asymmetrical quantization: only the routed MoE
experts are quantized, up/gate at `IQ2_XXS`, down at `Q2_K`. They are the
majority of all the model space: the other components (shared experts,
projections, routing) are left untouched to guarantee quality.

Download a model:

```sh
ds4f download q2   # 128 GB RAM machines (~81 GB on disk)
ds4f download q4   # >= 256 GB RAM machines (~153 GB on disk)
ds4f download mtp  # optional speculative decoding component (~3.5 GB)
```

The downloader uses `hf_hub` against
`https://huggingface.co/antirez/deepseek-v4-gguf`, reuses the Hugging Face
cache for resumable downloads, exposes the selected file under `./gguf/`, and
updates `./ds4flash.gguf` to point at the selected q2/q4 model. Authentication
is optional for public downloads, but `--token TOKEN`, `HF_TOKEN`, or the local
Hugging Face token cache are used when present.

The MTP component is optional and must be enabled explicitly with `--mtp`.
It is useful only for greedy decoding, currently uses a confidence gate
(`--mtp-margin`) to avoid slow partial accepts, and should be treated as an
experimental slight-speedup path.

## Build

Build `ds4f`:

```sh
cargo build --release --bin ds4f
```

Or for a quick development build:

```sh
cargo build --bin ds4f
```

The build compiles `ds4.c` (engine core, tokenizer, weight loading) and
`ds4_metal.m` (Metal kernel dispatch) into a static library. The Metal shaders
in `metal/*.metal` are compiled at runtime by Metal. No other external
dependencies are needed beyond the macOS SDK.

## Speed

These are single-run Metal numbers with `--ctx 32768`, `--nothink`, greedy
decoding, and `-n 256`. The short prompt is a normal small Italian story
prompt. The long prompts exercise chunked prefill plus long-context decode.
Q4 requires the larger-memory machine class, so M3 Max Q4 numbers are `N/A`.

| Machine | Quant | Prompt | Prefill | Generation |
| --- | ---: | ---: | ---: | ---: |
| MacBook Pro M3 Max, 128 GB | q2 | short | 58.52 t/s | 26.68 t/s |
| MacBook Pro M3 Max, 128 GB | q2 | 11709 tokens | 250.11 t/s | 21.47 t/s |
| MacBook Pro M3 Max, 128 GB | q4 | short | N/A | N/A |
| MacBook Pro M3 Max, 128 GB | q4 | long | N/A | N/A |
| Mac Studio M3 Ultra, 512 GB | q2 | short | 84.43 t/s | 36.86 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | 11709 tokens | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | short | 78.95 t/s | 35.50 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | 12018 tokens | 448.82 t/s | 26.62 t/s |

## CLI — `ds4f run`

One-shot prompt:

```sh
ds4f run -p "Explain Redis streams in one paragraph."
```

No `-p` starts the interactive REPL:

```sh
ds4f run
ds4f>
```

The interactive CLI is a real multi-turn DS4 chat. It keeps the rendered chat
transcript and the live Metal KV checkpoint, so each turn extends the previous
conversation. Useful commands are `/help`, `/think`, `/think-max`, `/nothink`,
`/ctx N`, `/read FILE`, and `/quit`. Ctrl+C interrupts the current generation
and returns to `ds4>`.

The CLI defaults to thinking mode. Use `/nothink` or `--nothink` for direct
answers.

Selected flags:

```
-m, --model PATH        Model GGUF (default: ds4flash.gguf)
    --mtp PATH          MTP speculative decoding GGUF
    --mtp-draft N       Draft tokens per step (default: 1)
-c, --ctx N             Context size in tokens (default: 32768)
-p, --prompt TEXT       One-shot prompt
    --prompt-file PATH  One-shot prompt from file
-s, --system TEXT       System prompt (default: "You are a helpful assistant")
-n, --tokens N          Maximum tokens to generate (default: 50000)
    --temp F            Temperature (default: 1.0)
    --top-p F           Top-p sampling (default: 1.0)
    --top-k N           Top-k sampling (default: 0 = disabled)
    --min-p F           Min-p sampling (default: 0.0)
    --seed N            RNG seed
    --nothink           Disable thinking mode
    --think             Enable thinking mode (default)
    --think-max         Maximum thinking budget
    --inspect           Print model summary and exit
    --dump-tokens       Print tokenized prompt and exit
    --warm-weights      Pre-warm weight pages before inference
    --quality           Quality mode (slower but slightly better)
```

## Server — `ds4f serve`

Start a local OpenAI/Anthropic-compatible server:

```sh
ds4f serve --ctx 100000
```

The server is Metal-only. Each request creates a fresh session; there is no
shared mutable KV state between requests. Request inference is serialized
through a single Mutex-protected engine.

Supported endpoints:

- `GET /v1/models`
- `GET /v1/models/deepseek-v4-flash`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/messages`

`/v1/chat/completions` accepts the usual OpenAI-style `messages`,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`seed`, and `stream`. The `thinking` field controls thinking mode.

`/v1/messages` is the Anthropic-compatible endpoint. It accepts `system`,
`messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, and
`thinking` controls.

Both endpoints support SSE streaming.

Minimal OpenAI example:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

Selected flags:

```
-m, --model PATH    Model GGUF (default: ds4flash.gguf)
    --mtp PATH      MTP speculative decoding GGUF
-c, --ctx N         Context size in tokens (default: 32768)
-n, --tokens N      Default max tokens per request (default: 393216)
    --host ADDR     Listen address (default: 127.0.0.1)
    --port N        Listen port (default: 8000)
    --warm-weights  Pre-warm weight pages before serving
    --quality       Quality mode
```

### Agent Client Configuration

`ds4f serve` is compatible with local coding agents that speak OpenAI chat
completions. Configure the context limit to match the `--ctx` value:

For **opencode**, add to `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ds4": {
      "name": "ds4 (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (ds4 local)",
          "limit": {
            "context": 100000,
            "output": 384000
          }
        }
      }
    }
  },
  "agent": {
    "ds4": {
      "description": "DeepSeek V4 Flash served by local ds4f serve",
      "model": "ds4/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

For **Pi**, add to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ds4": {
      "name": "ds4 local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dsv4-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "deepseek",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "deepseek-v4-flash",
          "name": "DeepSeek V4 Flash (ds4 local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 100000,
          "maxTokens": 384000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

## Testing

Validation uses official logits to check the engine output at the token level:

```sh
cargo test
```

The test vectors live under `tests/test-vectors/`. Fetch the official vectors with:

```sh
python3 tests/test-vectors/fetch_official_vectors.py
```

## Project Structure

```
ds4.c              Engine core: weight loading, tokenizer, MTP, GGUF mmap
ds4.h              Public C API
ds4_metal.m        Metal kernel dispatch
ds4_metal.h        Metal dispatch header
metal/*.metal      Metal compute shaders
src/lib.rs         Rust library: session, inference loop, sampling, chat prompt rendering
src/bin/ds4f.rs    Unified CLI: ds4f run / ds4f serve / ds4f download
src/ffi.rs         Rust FFI bindings for ds4.c and Metal
src/gguf.rs        Rust GGUF parser
build.rs           Compiles ds4.c and ds4_metal.m into libds4ffi.a
```
