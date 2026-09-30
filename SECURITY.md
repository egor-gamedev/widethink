# Security policy

## Supported versions

widethink is in alpha; only the latest release receives fixes.

## Reporting a vulnerability

Please do **not** open a public issue. Use GitHub's private vulnerability
reporting: *Security → Report a vulnerability* on the repository page. You will
get a reply within a week; once a fix is released, the report is credited unless
you prefer otherwise.

## What to keep in mind when using widethink

- **Your context goes to your model provider.** Project files passed as context
  are sent to the API you configure. For code that must not leave your machine,
  use a local model behind an OpenAI-compatible server.
- **Recordings and results contain prompts and answers.** Files written with
  `--record` and `--out` may include private code; the repository's
  `.gitignore` excludes `runs/`, `results/` and `*.recording.jsonl`. Review before
  sharing.
- **Model output is untrusted.** widethink never executes model output, but
  answers may contain code or commands; review them like any other suggestion.
- **Keys** are read by the provider SDKs from the environment. Never commit a
  `.env` file.
