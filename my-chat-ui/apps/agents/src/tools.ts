import * as z from "zod";
import { tool } from "@langchain/core/tools";
import { CohereClientV2 } from 'cohere-ai';
import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-large",
    apiKey: process.env.OPENAI_API_KEY,
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
    collectionName: "dance-contract",
    contentPayloadKey: "page_content",
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