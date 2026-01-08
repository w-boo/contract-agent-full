import { createAgent } from "langchain";
import { ChatOpenAI } from "@langchain/openai";
import { retrieveTool } from "./tools.js";
import { MemorySaver } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const model = new ChatOpenAI({
    model: "gpt-5-mini",
    temperature: 1,
});

const agent = createAgent({
    model,
    tools: [retrieveTool],
    checkpointer,
    systemPrompt: "You are a helpful assistant. Be concise and accurate.",
});

export { agent };