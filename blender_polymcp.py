#!/usr/bin/env python3
"""Blender MCP Chat - Come filesystem MCP"""
import asyncio
import sys
from pathlib import Path
from polymcp.polyagent import UnifiedPolyAgent, OllamaProvider

async def main():
    llm = OllamaProvider(model="gpt-oss:20b", temperature=0.1)
    
    # Invece di stdio, usiamo HTTP endpoint
    mcp_servers = ["http://localhost:8000/mcp"]
    
    agent = UnifiedPolyAgent(
        llm_provider=llm, 
        mcp_servers=mcp_servers,  
        verbose=True
    )
    
    async with agent:
        print("\n✅ Blender MCP Server connesso!\n")
        
        # Chat loop
        while True:
            user_input = input("\n🎨 Tu: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Arrivederci!")
                break
            
            result = await agent.run_async(user_input, max_steps=5)
            print(f"\n🤖 Blender: {result}")

if __name__ == "__main__":
    asyncio.run(main())