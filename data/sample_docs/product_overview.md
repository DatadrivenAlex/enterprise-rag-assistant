# Helios Insights Product Overview

Helios Insights is the internal analytics assistant for Helios Analytics. Finance, support, and operations use it to ask questions about company policy and product documentation. Answers must cite the source document they came from.

## What the assistant covers

The first release answers questions over markdown and plain-text documents that the knowledge team has indexed. Typical questions cover vacation rules, security duties, retention periods, and how Helios Insights itself is meant to be used. The assistant does not provide legal advice, and it does not change records in other systems.

## How answers are produced

The service retrieves a small set of relevant passages, then writes an answer grounded in those passages. Every answer includes a citation back to the source file and the chunk that supported it. If the indexed documents do not contain the answer, the assistant says so instead of guessing.

## Operating boundaries

Indexed documents stay in the company environment. The default deployment uses a local vector index and an offline extractive answerer, so a demo can run without an external model key. A deployment can switch the chat and embedding providers to an OpenAI-compatible API when one is approved.

## Users

Finance asks about expense timing and retention of analytics records. Support asks about ticket retention and incident reporting. Operations asks about remote-work limits and production access rules. New hires are pointed at this corpus during their first week.
