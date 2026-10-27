# Security policy

## Reporting

If you believe you found a security issue in this project, open a private security advisory on [github.com/DatadrivenAlex/enterprise-rag-assistant](https://github.com/DatadrivenAlex/enterprise-rag-assistant). Please include the version, the request you sent, and what the service returned. Do not open a public issue for an unfixed vulnerability.

This is a portfolio service. It has no production users and no bounty program.

## Threat model

The assistant answers questions from documents the operator indexes. Those documents, and the questions, are untrusted. The risks this codebase is built to reduce are:

- A question that tries to override the assistant instructions or extract the system prompt.
- A retrieved passage that contains the same kind of override and is copied into the prompt.
- An answer that leaks the internal canary planted in the system prompt.
- An ingest path that escapes the configured data directory.
- An optional API key that is compared in constant time when `RAG_API_KEY` is set.
- A model or embedding error that would otherwise include request headers in the API response.

The service does not authenticate users on its own when `RAG_API_KEY` is empty. That is acceptable for a local demo and is not acceptable on a shared network. Put the API behind a gateway, set `RAG_API_KEY`, and restrict `RAG_CORS_ORIGINS` before anyone else can reach it.

## Controls in this repository

| Control | Where it lives |
| --- | --- |
| Question screening for instruction-override phrases, fake role changes, and fake system tags | `security.py` |
| The same screening on ingest and again on retrieval, with matching chunks dropped | `rag_pipeline.py` |
| Retrieved text wrapped as data, with prompt delimiters stripped out of the passage | `rag_pipeline.py` |
| Canary in the system prompt; answers that echo it are replaced with a refusal | `security.py` |
| Ingest paths resolved under `RAG_DATA_DIR` | `document_loader.py` |
| Upload type and size limits | `routes.py` |
| Optional `X-API-Key` on `/api/v1/*` | `routes.py` |
| Citations returned with every grounded answer | `rag_pipeline.py` |
| Offline lexical evaluation of faithfulness, relevancy, context precision, and citation support | `evaluator.py` |

Screening looks for attack-shaped phrases. It is not a guarantee against every paraphrase. Treat it as one layer, and keep the model instruction that context is data rather than instructions.

## Out of scope

Authentication beyond the optional API key, per-user document permissions, malware analysis, and PDF or office-document parsing are not implemented. Hash embeddings are for offline development. Use an approved embedding model when the corpus is large or the wording of questions will not overlap the documents.

## Supported use

The default configuration never calls an external model. Tests are written to run without an API key. If you set `RAG_CHAT_PROVIDER=openai` or `RAG_EMBEDDING_PROVIDER=openai`, you are sending questions and document text to that endpoint. Review that choice before indexing anything confidential.

## Operational notes

Run a single Uvicorn worker. The Chroma and FAISS adapters in this process are not a multi-writer cluster. Keep `.env` out of git. The example file ships with empty secrets on purpose.
