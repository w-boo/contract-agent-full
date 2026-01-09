import * as z from "zod";
import { tool } from "@langchain/core/tools";
import { CohereClientV2 } from 'cohere-ai';
import { QdrantClient } from "@qdrant/js-client-rest";

async function embedQuery(text: string): Promise<number[]> {
    const response = await fetch("https://api.jina.ai/v1/embeddings", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${process.env.JINA_API_KEY}`,
        },
        body: JSON.stringify({
            model: "jina-embeddings-v4",
            task: "retrieval.query",
            input: [text],
        }),
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Jina API error: ${response.status} ${errorText}`);
    }

    const data = await response.json();
    return data.data[0].embedding;
}

const qdrantClient = new QdrantClient({
    url: process.env.QDRANT_URL!,
    apiKey: process.env.QDRANT_API_KEY!,
});

const cohere = new CohereClientV2({
    token: process.env.COHERE_API_KEY
});

export const retrieveTool = tool(
    async ({ query }: { query: string }) => {
        const queryEmbedding = await embedQuery(query);

        const queryResult = await qdrantClient.query("MyCollection", {
            query: queryEmbedding,
            limit: 40,
            with_payload: true,
        });

        const documents = (queryResult.points || [])
            .map(point => {
                const payload = point.payload || {};
                const text = typeof payload.text === 'string' ? payload.text : "";

                const { text: _, ...metadata } = payload;

                return {
                    text: text,
                    metadata: metadata,
                };
            })
            .filter(doc => doc.text && doc.text.trim().length > 0);

        if (documents.length === 0) {
            return "No relevant documents found for the query.";
        }

        const texts = documents.map(doc => doc.text);
        const rerankResponse = await cohere.rerank({
            model: "rerank-v4.0-fast",
            query: query,
            documents: texts,
            topN: Math.min(5, texts.length),
        });

        const topChunks = rerankResponse.results.map(
            result => {
                const doc = documents[result.index];
                const score = result.relevanceScore;

                const metadataStr = Object.keys(doc.metadata).length > 0
                    ? `\n[Metadata: ${JSON.stringify(doc.metadata, null, 2)}]`
                    : '';

                return `[Relevancy Score: ${score.toFixed(4)}]${metadataStr}\n${doc.text}`;
            }
        );

        return topChunks.join("\n\n-----\n\n");
    },
    {
        name: "retrieve",
        description: "Retrieve relevant contract/choreography chunks using vector search and Cohere cross-encoder reranking.",
        schema: z.object({
            query: z.string().describe("The search query to retrieve relevant chunks"),
        }),
    }
);