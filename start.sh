./ds4-server -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
--mtp ./gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
--mtp-draft 3 \
--ctx 1000000 \
--threads 16 \
--metal \
--kv-disk-dir ./kv_cache \
--kv-disk-space-mb 10240

