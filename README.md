# Blender MCP Server 🎨

**Production-ready Model Context Protocol (MCP) server for complete Blender control via API**

This addon transforms Blender into an MCP server, exposing 100+ powerful tools that can be orchestrated by AI agents through HTTP endpoints. Perfect for AI-driven 3D workflows, automation, and creative experimentation.

## ✨ Features

- **🚀 100+ Production-Ready Tools** - Complete control over Blender's features:
  - Object creation, manipulation, and transformation
  - Material and shader system management
  - Animation and keyframe control
  - Camera and lighting setup
  - Modifiers and constraints
  - Physics simulations (rigid body, cloth, fluid)
  - Geometry nodes and procedural generation
  - File import/export operations
  - Scene optimization and batch operations

- **🔒 Thread-Safe Execution** - Enterprise-grade queue system for safe concurrent operations
- **📦 Auto-Install Dependencies** - Automatically installs required packages on first run
- **🎮 Simple UI Panel** - Start/stop server with one click from Blender's N-panel
- **📊 Real-time Monitoring** - Track operations, statistics, and server status
- **🔧 Production-Ready** - Comprehensive error handling, logging, and caching

## 🚀 Quick Start

### Installation

1. Download `blender_mcp.py`
2. Open Blender
3. Go to **Edit → Preferences → Add-ons**
4. Click the **dropdown arrow** (⌄) next to the search bar
5. Select **Install from Disk...**
6. Choose the downloaded `blender_mcp.py` file
7. Enable the addon by checking the box next to "MCP Complete Server for Blender"

### Starting the Server

1. Press **N** in the 3D Viewport to open the sidebar
2. Navigate to the **MCP Server** tab
3. Click **Start Server**
4. Server will start on `http://localhost:8000`

The addon will automatically install required dependencies on first run (FastAPI, Uvicorn, Pydantic, etc.).

## 🤖 Using with PolyMCP

This MCP server is designed to work seamlessly with **[PolyMCP](https://github.com/llm-use/Polymcp)** - a powerful framework for orchestrating MCP servers with AI agents.

### Example: AI-Controlled Blender

```python
#!/usr/bin/env python3
import asyncio
from polymcp.polyagent import UnifiedPolyAgent, OllamaProvider

async def main():
    # Initialize your LLM provider
    llm = OllamaProvider(model="gpt-oss:20b", temperature=0.1)
    
    # Connect to Blender MCP server
    agent = UnifiedPolyAgent(
        llm_provider=llm, 
        mcp_servers=["http://localhost:8000/mcp"],  
        verbose=True
    )
    
    async with agent:
        print("✅ Blender MCP Server connected!\n")
        
        # Chat with your AI to control Blender
        while True:
            user_input = input("\n🎨 You: ")
            
            if user_input.lower() in ['exit', 'quit']:
                break
            
            result = await agent.run_async(user_input, max_steps=5)
            print(f"\n🤖 Blender: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example Commands

Once connected, you can ask the AI agent to:

- *"Create a cube at position (0, 0, 0) with size 2"*
- *"Add a red metallic material to the selected object"*
- *"Create a camera looking at the origin"*
- *"Set up a simple lighting scene with 3 lights"*
- *"Add a fluid simulation to the cube"*
- *"Export the scene as an FBX file"*
- *"Create an animation rotating the object 360 degrees over 100 frames"*

**That's it!** PolyMCP handles all the complexity of:
- Tool discovery and selection
- Multi-step task planning
- Error handling and retries
- State management across operations

This makes it incredibly simple to build AI-powered Blender automation tools!

## 📡 API Endpoints

Once the server is running, you can access:

- **API Documentation**: `http://localhost:8000/docs`
- **List All Tools**: `http://localhost:8000/mcp/list_tools`
- **Invoke Tool**: `POST http://localhost:8000/mcp/invoke/{tool_name}`

## 🛠️ Available Tool Categories

<details>
<summary>View all tool categories (100+ tools)</summary>

- **Object Operations**: Create, delete, duplicate, select objects
- **Transformations**: Move, rotate, scale, apply transforms
- **Materials & Shading**: Create materials, add textures, setup shader nodes
- **Modeling**: Add modifiers, boolean operations, mesh editing
- **Animation**: Keyframes, timeline control, NLA editor
- **Camera & Lighting**: Camera setup, light creation, HDRI environments
- **Rendering**: Render settings, output configuration, rendering
- **Physics**: Rigid body, cloth, fluid simulations
- **Geometry Nodes**: Procedural generation, node tree creation
- **File Operations**: Import/export various formats (FBX, OBJ, USD, etc.)
- **Scene Management**: Scene info, cleanup, optimization
- **Batch Operations**: Multi-object creation and transformation
- **Advanced**: Particle systems, force fields, grease pencil

</details>

## 🔧 Configuration

You can customize the server by editing the `Config` class in `blender_mcp.py`:

```python
class Config:
    HOST = "0.0.0.0"           # Server host
    PORT = 8000                # Server port
    AUTO_INSTALL_PACKAGES = True  # Auto-install dependencies
    THREAD_SAFE_OPERATIONS = True  # Enable thread-safe execution
    ENABLE_CACHING = True      # Enable result caching
```

## 📋 Requirements

The addon automatically installs these dependencies:

- FastAPI
- Uvicorn
- Pydantic
- docstring-parser
- NumPy
- PolyMCP toolkit (included)

**Blender Version**: 3.0.0 or higher

## 🐛 Troubleshooting

**Server won't start?**
- Check Blender's System Console for error messages (Window → Toggle System Console)
- Ensure port 8000 is not already in use
- Try restarting Blender after installation

**Dependencies not installing?**
- Manually install packages using Blender's Python:
  ```bash
  /path/to/blender/python -m pip install fastapi uvicorn pydantic docstring-parser numpy
  ```

**Can't find MCP Server panel?**
- Press **N** in the 3D Viewport
- Look for "MCP Server" tab in the sidebar
- Make sure the addon is enabled in Preferences

## 🤝 Contributing

Contributions are welcome! This project demonstrates how simple it is to create powerful MCP servers with PolyMCP.

## 📝 License

MIT License - See LICENSE file for details

## 🔗 Related Projects

- **[PolyMCP](https://github.com/llm-use/Polymcp)** - Universal MCP Agent & Toolkit for Intelligent Tool Orchestration
- **[Model Context Protocol](https://modelcontextprotocol.io/)** - Open protocol for tool integration with LLMs

## 💡 Why This Project?

This MCP server was created to demonstrate how **incredibly simple** PolyMCP makes it to build AI-powered tools. With just a few lines of code, you can:

1. Expose complex Blender functionality as MCP tools
2. Let AI agents discover and orchestrate these tools
3. Build natural language interfaces for 3D creation

No complex prompting, no manual tool selection, no state management - PolyMCP handles it all!

---

**Built with ❤️ using [PolyMCP](https://github.com/llm-use/Polymcp)**

*Star ⭐ this repo if you find it useful!*
