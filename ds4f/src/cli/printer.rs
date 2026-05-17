//! Token printer with think-mode tag handling.
//!
//! Mirrors the `token_printer` in ds4_cli.c: buffers partial bytes,
//! strips `<think>` / `</think>` tags from output, and applies color
//! to think content when writing to a terminal.

use std::io::{self, Write};

const THINK_OPEN: &[u8] = b"<think>";
const THINK_CLOSE: &[u8] = b"</think>";

pub struct TokenPrinter {
    out: Box<dyn Write>,
    use_color: bool,
    in_think: bool,
    pending: Vec<u8>,
}

impl TokenPrinter {
    pub fn new(out: Box<dyn Write>, use_color: bool) -> Self {
        Self { out, use_color, in_think: false, pending: Vec::with_capacity(16) }
    }

    /// Process a chunk of token text, flushing complete segments.
    /// Tags are stripped (never printed) and act as color-state toggles.
    pub fn process(&mut self, text: &[u8], finish: bool) -> io::Result<()> {
        self.pending.extend_from_slice(text);
        self.drain(finish)
    }

    /// Signal end of generation — flush any remaining pending bytes.
    pub fn finish(&mut self) -> io::Result<()> {
        self.drain(true)?;
        if self.use_color && self.in_think {
            write!(self.out, "\x1b[0m")?;
            self.in_think = false;
        }
        writeln!(self.out)?;
        Ok(())
    }

    fn drain(&mut self, finish: bool) -> io::Result<()> {
        loop {
            // Find the next tag start, or decide to flush
            let (flush_end, skip_len, entering_think, leaving_think) =
                if let Some(pos) = find_subsequence(&self.pending, THINK_OPEN) {
                    (pos, THINK_OPEN.len(), true, false)
                } else if let Some(pos) = find_subsequence(&self.pending, THINK_CLOSE) {
                    (pos, THINK_CLOSE.len(), false, true)
                } else if finish {
                    (self.pending.len(), 0, false, false)
                } else {
                    break; // nothing to flush, keep buffering
                };

            // Write everything before the tag
            if flush_end > 0 {
                let chunk: Vec<u8> = self.pending.drain(..flush_end).collect();
                if self.use_color && self.in_think {
                    write!(self.out, "\x1b[90m")?; // grey
                    self.out.write_all(&chunk)?;
                    write!(self.out, "\x1b[0m")?;
                } else {
                    self.out.write_all(&chunk)?;
                }
            }

            // Skip the tag bytes (never print them)
            if skip_len > 0 {
                self.pending.drain(..skip_len);
                if entering_think {
                    self.in_think = true;
                } else if leaving_think {
                    self.in_think = false;
                }
            }

            if self.pending.is_empty() {
                break;
            }
        }

        self.out.flush()
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
