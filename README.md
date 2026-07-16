# NagrikAI

AI-powered Indian immigration assistant using a RAG pipeline over official sources.

## CLI Usage

```bash
nagrik-ai [OPTIONS] COMMAND [ARGS]...
```

### Commands

Examples:

```bash
nagrik-ai crawl sites                     # crawl all sites at depth 1
nagrik-ai crawl sites --depth 2           # crawl deeper
nagrik-ai crawl sites --depth 0           # unlimited depth
nagrik-ai crawl sites --manage            # skip URLs already crawled
nagrik-ai crawl sites -d 2 -m             # depth 2 + management mode
```

### `nagrik-ai parse all`

```bash
nagrik-ai parse all [OPTIONS]
```

| Option          | Short | Type   | Default   | Description       |
| --------------- | ----- | ------ | --------- | ----------------- |
| `--content-dir` | `-d`  | `PATH` | `content` | Content directory |

### `nagrik-ai vectorize run`

```bash
nagrik-ai vectorize run [OPTIONS]
```

| Option            | Short | Type   | Default     | Description                |
| ----------------- | ----- | ------ | ----------- | -------------------------- |
| `--content-dir`   | `-d`  | `PATH` | `content`   | Content directory          |
| `--persist-dir`   | `-p`  | `PATH` | `chroma_db` | ChromaDB persist directory |
| `--chunk-size`    | `-s`  | `INT`  | `512`       | Chunk size                 |
| `--chunk-overlap` | `-o`  | `INT`  | `64`        | Chunk overlap              |

### `nagrik-ai app-command`

```bash
nagrik-ai app-command [OPTIONS]
```

| Option          | Short | Type   | Default     | Description                |
| --------------- | ----- | ------ | ----------- | -------------------------- |
| `--persist-dir` | `-p`  | `PATH` | `chroma_db` | ChromaDB persist directory |
| `--share`       | —     | flag   | `False`     | Create a public link       |
| `--port`        | —     | `INT`  | `7860`      | Port to run on             |
