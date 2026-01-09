import * as z from "zod";
import { tool } from "@langchain/core/tools";
import { CohereClientV2 } from 'cohere-ai';
import { QdrantVectorStore } from "@langchain/qdrant";
import { Embeddings } from "@langchain/core/embeddings";

class JinaEmbeddingsV4 extends Embeddings {
    private apiKey: string;
    private model: string;

    constructor(config: { apiKey: string; model?: string }) {
        super({});
        this.apiKey = config.apiKey;
        this.model = config.model || "jina-embeddings-v4";
    }

    async embedDocuments(texts: string[]): Promise<number[][]> {
        const response = await fetch("https://api.jina.ai/v1/embeddings", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${this.apiKey}`,
            },
            body: JSON.stringify({
                model: this.model,
                task: "retrieval.passage",
                input: texts,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Jina API error: ${response.status} ${errorText}`);
        }

        const data = await response.json();
        return data.data.map((item: any) => item.embedding);
    }

    async embedQuery(text: string): Promise<number[]> {
        const response = await fetch("https://api.jina.ai/v1/embeddings", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${this.apiKey}`,
            },
            body: JSON.stringify({
                model: this.model,
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
}

const embeddings = new JinaEmbeddingsV4({
    apiKey: process.env.JINA_API_KEY!,
    model: "jina-embeddings-v4",
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
    collectionName: "MyCollection",
    contentPayloadKey: "text",
});

const cohere = new CohereClientV2({
    token: process.env.COHERE_API_KEY
});

const vectorRetriever = await vectorStore.asRetriever({
    searchKwargs: {
        fetchK: 40,
    },
});

export const retrieveTool = tool(
    async ({ query }: { query: string }) => {
        const docs = await vectorRetriever.invoke(query);

        const texts = docs
            .map(doc => doc.pageContent)
            .filter(text => text && text.trim().length > 0);

        if (texts.length === 0) {
            return "No relevant documents found for the query.";
        }

        const rerankResponse = await cohere.rerank({
            model: "rerank-v4.0-fast",
            query: query,
            documents: texts,
            topN: Math.min(5, texts.length),
        });

        const topChunks = rerankResponse.results.map(
            result => texts[result.index]
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