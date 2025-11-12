"""
BLENDER MCP COMPLETE SERVER - PRODUCTION VERSION
Enterprise-grade MCP server for complete Blender control via API
Version: 3.0.0
"""

import bpy
import sys
import subprocess
import os
import importlib
import math
import queue
import time
import functools
import uuid
import traceback
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from functools import lru_cache

bl_info = {
    "name": "MCP Complete Server for Blender - Production",
    "author": "MCP Blender Team",
    "version": (3, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > N-Panel > MCP Server",
    "description": "Production-ready MCP server for complete Blender control via API",
    "category": "Development",
}

# ============================================
# 🔧 CONFIGURATION & LOGGING
# ============================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [Blender MCP] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Server configuration"""
    HOST = "0.0.0.0"
    PORT = 8000
    QUEUE_TIMEOUT = 30.0
    QUEUE_CHECK_INTERVAL = 0.01
    MAX_QUEUE_SIZE = 1000
    THREAD_SAFE_OPERATIONS = True
    AUTO_INSTALL_PACKAGES = True
    POLYMCP_PATH = r'C:\Users\vincenzo\Desktop\polymcp\polymcp'
    ENABLE_CACHING = True
    CACHE_SIZE = 256

# ============================================
# 📦 PACKAGE MANAGEMENT
# ============================================

def check_and_install_packages():
    """Check and install required packages with proper error handling"""
    required_packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn[standard]',
        'pydantic': 'pydantic',
        'docstring_parser': 'docstring-parser',
        'numpy': 'numpy',
    }
    
    python_exe = sys.executable
    missing_packages = []
    
    logger.info("Checking required packages...")
    
    for module_name, install_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            logger.info(f"✓ {module_name} found")
        except ImportError:
            missing_packages.append(install_name)
            logger.warning(f"✗ {module_name} missing")
    
    if missing_packages and Config.AUTO_INSTALL_PACKAGES:
        logger.info("Installing missing packages...")
        
        try:
            # Ensure pip is available and updated
            subprocess.check_call([python_exe, "-m", "ensurepip"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            
            # Install missing packages
            for package in missing_packages:
                logger.info(f"Installing {package}...")
                try:
                    subprocess.check_call([python_exe, "-m", "pip", "install", package],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                    logger.info(f"✓ {package} installed successfully")
                except subprocess.CalledProcessError as e:
                    logger.error(f"✗ Failed to install {package}: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Package installation failed: {e}")
            return False
    
    return True

# Check packages on import
packages_ok = check_and_install_packages()

if packages_ok:
    try:
        # Add polymcp path
        if Config.POLYMCP_PATH not in sys.path:
            sys.path.append(Config.POLYMCP_PATH)
            
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from polymcp_toolkit import expose_tools
        import numpy as np
        logger.info("✓ All packages loaded successfully")
    except ImportError as e:
        logger.error(f"Import error after installation: {e}")
        logger.info("Please restart Blender and try again")
else:
    logger.error("Failed to install required packages")

import bmesh
import mathutils
import threading
import base64
import tempfile
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor

# ============================================
# 🔄 THREAD-SAFE EXECUTION SYSTEM
# ============================================

class ThreadSafeExecutor:
    """Production-grade thread-safe execution system for Blender operations"""
    
    def __init__(self):
        self.execution_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.result_store = {}
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    def start(self):
        """Start the queue processor"""
        if not self.is_running:
            if not bpy.app.timers.is_registered(self._process_queue):
                bpy.app.timers.register(self._process_queue)
                self.is_running = True
                logger.info("Thread-safe executor started")
    
    def stop(self):
        """Stop the queue processor"""
        if self.is_running:
            if bpy.app.timers.is_registered(self._process_queue):
                bpy.app.timers.unregister(self._process_queue)
                self.is_running = False
                logger.info("Thread-safe executor stopped")
    
    def _process_queue(self):
        """Process pending operations in the main thread"""
        try:
            while not self.execution_queue.empty():
                try:
                    request = self.execution_queue.get_nowait()
                    request_id = request['id']
                    func = request['func']
                    args = request['args']
                    kwargs = request['kwargs']
                    
                    try:
                        # Execute in main thread
                        result = func(*args, **kwargs)
                        self.result_store[request_id] = {
                            'status': 'success',
                            'result': result,
                            'timestamp': time.time()
                        }
                    except Exception as e:
                        error_details = {
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'function': func.__name__,
                            'args': str(args)[:200],
                            'kwargs': str(kwargs)[:200]
                        }
                        self.result_store[request_id] = {
                            'status': 'error',
                            'error': error_details,
                            'timestamp': time.time()
                        }
                        logger.error(f"Error executing {func.__name__}: {e}")
                        
                except queue.Empty:
                    break
                    
        except Exception as e:
            logger.error(f"Queue processor error: {e}")
        
        # Clean old results (older than 5 minutes)
        current_time = time.time()
        expired_ids = [
            rid for rid, data in self.result_store.items()
            if current_time - data['timestamp'] > 300
        ]
        for rid in expired_ids:
            del self.result_store[rid]
        
        return Config.QUEUE_CHECK_INTERVAL
    
    def execute(self, func, *args, **kwargs):
        """Execute a function in the main thread and return the result"""
        request_id = str(uuid.uuid4())
        
        # Add to queue
        try:
            self.execution_queue.put({
                'id': request_id,
                'func': func,
                'args': args,
                'kwargs': kwargs
            }, timeout=1.0)
        except queue.Full:
            raise RuntimeError("Execution queue is full")
        
        # Wait for result
        start_time = time.time()
        while time.time() - start_time < Config.QUEUE_TIMEOUT:
            if request_id in self.result_store:
                result_data = self.result_store.pop(request_id)
                
                if result_data['status'] == 'success':
                    return result_data['result']
                else:
                    error_info = result_data['error']
                    raise RuntimeError(f"Execution failed: {error_info['error']}")
            
            time.sleep(0.001)
        
        raise TimeoutError(f"Operation timed out after {Config.QUEUE_TIMEOUT}s")

# Global executor instance
thread_executor = ThreadSafeExecutor()

def thread_safe(func):
    """Decorator to make functions thread-safe"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if Config.THREAD_SAFE_OPERATIONS:
            return thread_executor.execute(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

@thread_safe
def capture_viewport_image(
    camera_name: Optional[str] = None,
    resolution: Tuple[int, int] = (512, 512),
    samples: int = 8,
    return_base64: bool = True,
    include_overlays: bool = False
) -> Dict[str, Any]:
    """
    Capture viewport for VLM analysis.
    """
    # Store original settings
    scene = bpy.context.scene
    original_res_x = scene.render.resolution_x
    original_res_y = scene.render.resolution_y
    original_samples = scene.eevee.taa_render_samples
    
    # Set quick render settings
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = samples
    
    # Set camera if specified
    if camera_name:
        cam = bpy.data.objects.get(camera_name)
        if cam:
            scene.camera = cam
    
    # Render to temp file
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    scene.render.filepath = tmp.name
    
    # Quick viewport render
    bpy.ops.render.opengl(write_still=True, view_context=True)
    
    # Restore settings
    scene.render.resolution_x = original_res_x
    scene.render.resolution_y = original_res_y
    scene.eevee.taa_render_samples = original_samples
    
    result = {
        "resolution": resolution,
        "filepath": tmp.name
    }
    
    if return_base64:
        with open(tmp.name, 'rb') as f:
            result["image_base64"] = base64.b64encode(f.read()).decode('utf-8')
        os.unlink(tmp.name)
    
    return result

@thread_safe
def analyze_spatial_layout() -> Dict[str, Any]:
    """
    Generate spatial description for VLM context.
    """
    objects = []
    for obj in bpy.data.objects[:20]:  # Limit to 20 for performance
        if obj.type in ['MESH', 'CURVE', 'FONT']:
            bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
            center = sum(bbox, mathutils.Vector()) / 8
            
            objects.append({
                "name": obj.name,
                "type": obj.type,
                "center": [round(c, 2) for c in center],
                "size": [round(d, 2) for d in obj.dimensions]
            })
    
    # Group by proximity
    groups = []
    for obj in objects[:5]:  # Analyze first 5
        nearby = []
        for other in objects:
            if obj["name"] != other["name"]:
                dist = (mathutils.Vector(obj["center"]) - mathutils.Vector(other["center"])).length
                if dist < 3:
                    nearby.append(other["name"])
        if nearby:
            groups.append(f"{obj['name']} near {nearby[0]}")
    
    return {
        "object_count": len(bpy.data.objects),
        "main_objects": objects[:10],
        "spatial_groups": groups[:5],
        "viewport_center": [0, 0, 0]
    }

@thread_safe
def verify_last_operation(
    expected_result: Dict[str, Any],
    capture_image: bool = True
) -> Dict[str, Any]:
    """
    Verify last operation for VLM feedback loop.
    """
    verification = {
        "success": True,
        "issues": [],
        "suggestions": []
    }
    
    # Check if expected object exists
    if "object_name" in expected_result:
        obj = bpy.data.objects.get(expected_result["object_name"])
        if not obj:
            verification["success"] = False
            verification["issues"].append("Object not found")
            return verification
        
        # Check position if specified
        if "expected_location" in expected_result:
            expected = mathutils.Vector(expected_result["expected_location"])
            actual = obj.location
            distance = (expected - actual).length
            
            if distance > 0.5:
                verification["issues"].append(f"Position off by {distance:.2f}")
                verification["suggestions"].append(f"Move to {list(expected)}")
    
    # Capture viewport if requested
    if capture_image:
        verification["viewport"] = capture_viewport_image(resolution=(256, 256))
    
    # Add spatial context
    verification["spatial_context"] = analyze_spatial_layout()
    
    return verification

@thread_safe
def auto_arrange_objects(
    strategy: str = "grid",
    spacing: float = 2.0
) -> Dict[str, Any]:
    """
    Auto-arrange objects for better VLM visibility.
    """
    objects = [o for o in bpy.data.objects if o.type == 'MESH']
    
    if strategy == "grid":
        cols = int(math.sqrt(len(objects)))
        for i, obj in enumerate(objects):
            row = i // cols
            col = i % cols
            obj.location = [col * spacing, -row * spacing, 0]
    
    elif strategy == "circle":
        angle_step = 360 / len(objects) if objects else 1
        radius = spacing * len(objects) / (2 * math.pi)
        
        for i, obj in enumerate(objects):
            angle = math.radians(i * angle_step)
            obj.location = [
                math.cos(angle) * radius,
                math.sin(angle) * radius,
                0
            ]
    
    elif strategy == "line":
        for i, obj in enumerate(objects):
            obj.location = [i * spacing, 0, 0]
    
    return {
        "arranged": len(objects),
        "strategy": strategy,
        "spacing": spacing
    }

@thread_safe
def set_optimal_camera_for_all() -> Dict[str, Any]:
    """
    Position camera to see all objects (for VLM).
    """
    if not bpy.data.objects:
        return {"error": "No objects in scene"}
    
    # Calculate bounding box of all objects
    min_co = [float('inf')] * 3
    max_co = [float('-inf')] * 3
    
    for obj in bpy.data.objects:
        if obj.type in ['MESH', 'CURVE', 'FONT']:
            for corner in obj.bound_box:
                world_co = obj.matrix_world @ mathutils.Vector(corner)
                for i in range(3):
                    min_co[i] = min(min_co[i], world_co[i])
                    max_co[i] = max(max_co[i], world_co[i])
    
    center = [(min_co[i] + max_co[i]) / 2 for i in range(3)]
    size = [max_co[i] - min_co[i] for i in range(3)]
    distance = max(size) * 2
    
    # Create or get camera
    if not bpy.context.scene.camera:
        cam_data = bpy.data.cameras.new("VLM_Camera")
        cam_obj = bpy.data.objects.new("VLM_Camera", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
    else:
        cam_obj = bpy.context.scene.camera
    
    # Position camera
    cam_obj.location = [
        center[0] + distance * 0.7,
        center[1] - distance,
        center[2] + distance * 0.5
    ]
    
    # Point at center
    direction = mathutils.Vector(center) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    return {
        "camera": cam_obj.name,
        "target_center": center,
        "view_distance": distance
    }

# ============================================
# 🎆 PARTICLE SYSTEMS & PHYSICS
# ============================================

@thread_safe
def create_particle_system(
    object_name: str,
    particle_type: str = "HAIR",
    count: int = 1000,
    physics_type: str = "NEWTONIAN",
    emit_from: str = "FACE",
    lifetime: int = 50,
    gravity: float = 1.0,
    size: float = 0.05,
    random_size: float = 0.0,
    velocity_normal: float = 1.0,
    velocity_random: float = 0.5,
    hair_length: float = 0.2,
    hair_segments: int = 5,
    brownian_factor: float = 0.0,
    damping: float = 0.0,
    use_children: bool = False,
    child_count: int = 10,
    render_type: str = "PATH",
    material_slot: int = 1
) -> Dict[str, Any]:
    """
    Create advanced particle systems for hair, smoke, and effects.
    
    Args:
        object_name: Emitter object
        particle_type: HAIR or EMITTER
        count: Number of particles
        physics_type: NEWTONIAN, KEYED, BOIDS, FLUID, NO
        emit_from: VERT, FACE, VOLUME
        lifetime: Particle lifetime in frames
        gravity: Gravity influence
        size: Particle size
        random_size: Size randomization
        velocity_normal: Normal velocity
        velocity_random: Random velocity
        hair_length: Length for hair particles
        hair_segments: Segments for hair
        brownian_factor: Brownian motion
        damping: Velocity damping
        use_children: Enable child particles
        child_count: Number of children per particle
        render_type: PATH, OBJECT, COLLECTION, NONE
        material_slot: Material slot for particles
        
    Returns:
        Complete particle system configuration
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    # Add particle system modifier
    particle_mod = obj.modifiers.new(name="ParticleSystem", type='PARTICLE_SYSTEM')
    particle_system = obj.particle_systems[particle_mod.name]
    settings = particle_system.settings
    
    # Basic settings
    settings.type = particle_type
    settings.count = count
    settings.emit_from = emit_from
    settings.lifetime = lifetime
    settings.particle_size = size
    settings.size_random = random_size
    
    # Physics settings
    settings.physics_type = physics_type
    
    if physics_type == "NEWTONIAN":
        settings.mass = 1.0
        settings.effector_weights.gravity = gravity
        settings.normal_factor = velocity_normal
        settings.factor_random = velocity_random
        settings.brownian_factor = brownian_factor
        settings.damping = damping
        
    elif physics_type == "BOIDS":
        boids = settings.boids
        boids.use_flight = True
        boids.use_land = False
        boids.use_climb = False
        boids.air_speed_max = 10.0
        boids.air_speed_min = 5.0
        boids.air_acc_max = 0.5
        boids.bank = 1.0
        boids.pitch = 1.0
        
        # Add default boid rules
        rule = boids.states[0].rules.new(type='SEPARATE')
        rule.distance = 2.0
        rule = boids.states[0].rules.new(type='FLOCK')
        rule.distance = 10.0
        rule = boids.states[0].rules.new(type='FOLLOW_LEADER')
        rule.distance = 5.0
        
    elif physics_type == "FLUID":
        fluid = settings.fluid
        fluid.solver = 'DDR'
        fluid.stiffness = 1.0
        fluid.linear_viscosity = 1.0
        fluid.use_initial_rest_length = True
    
    # Hair specific settings
    if particle_type == "HAIR":
        settings.hair_length = hair_length
        settings.hair_step = hair_segments
        settings.root_radius = 1.0
        settings.tip_radius = 0.0
        
    # Children
    if use_children:
        settings.child_type = 'INTERPOLATED' if particle_type == "HAIR" else 'SIMPLE'
        settings.rendered_child_count = child_count
        settings.child_radius = 0.2
        settings.child_roundness = 1.0
    
    # Render settings
    settings.render_type = render_type
    settings.material_slot = material_slot
    
    if render_type == "PATH":
        settings.render_step = 3
        settings.path_start = 0.0
        settings.path_end = 1.0
    
    return {
        "object": object_name,
        "particle_system": particle_mod.name,
        "type": particle_type,
        "physics": physics_type,
        "count": count,
        "lifetime": lifetime,
        "emit_from": emit_from,
        "size": size,
        "children": {
            "enabled": use_children,
            "count": child_count if use_children else 0
        },
        "render_type": render_type,
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def add_force_field(
    field_type: str = "FORCE",
    strength: float = 1.0,
    location: Optional[List[float]] = None,
    flow: float = 0.0,
    noise: float = 0.0,
    falloff_type: str = "SPHERE",
    falloff_power: float = 2.0,
    max_distance: float = 0.0,
    min_distance: float = 0.0,
    use_radial_min: bool = False,
    use_radial_max: bool = True,
    name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add force fields for particle and physics simulations.
    
    Args:
        field_type: FORCE, WIND, VORTEX, MAGNETIC, HARMONIC, CHARGE, LENNARDJ, TEXTURE, GUIDE, BOID, TURBULENCE, DRAG, SMOKE
        strength: Field strength
        location: Field position
        flow: Inflow/outflow for fluid
        noise: Noise amount
        falloff_type: SPHERE, TUBE, CONE
        falloff_power: Power of falloff
        max_distance: Maximum effect distance
        min_distance: Minimum effect distance
        use_radial_min: Use minimum radial distance
        use_radial_max: Use maximum radial distance
        name: Force field name
        
    Returns:
        Force field configuration
    """
    location = location or [0, 0, 2]
    
    # Create empty for force field
    bpy.ops.object.empty_add(type='SPHERE', location=location)
    field_obj = bpy.context.active_object
    field_obj.name = name or f"ForceField_{field_type}"
    
    # Add force field
    field_obj.field.type = field_type
    field_obj.field.strength = strength
    field_obj.field.flow = flow
    field_obj.field.noise = noise
    field_obj.field.falloff_type = falloff_type
    field_obj.field.falloff_power = falloff_power
    field_obj.field.distance_max = max_distance
    field_obj.field.distance_min = min_distance
    field_obj.field.use_radial_min = use_radial_min
    field_obj.field.use_radial_max = use_radial_max
    
    # Type-specific settings
    if field_type == "VORTEX":
        field_obj.field.inflow = 5.0
    elif field_type == "TURBULENCE":
        field_obj.field.size = 1.0
        field_obj.field.flow = 0.5
    elif field_type == "HARMONIC":
        field_obj.field.harmonic_damping = 0.5
    
    return {
        "name": field_obj.name,
        "type": field_type,
        "strength": strength,
        "location": list(field_obj.location),
        "falloff": {
            "type": falloff_type,
            "power": falloff_power,
            "max_distance": max_distance,
            "min_distance": min_distance
        },
        "created_at": datetime.now().isoformat()
    }

# ============================================
# 🔷 GEOMETRY NODES
# ============================================

@thread_safe
def add_geometry_nodes(
    object_name: str,
    node_tree_name: Optional[str] = None,
    nodes_config: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Add and configure geometry nodes for procedural modeling.
    
    Args:
        object_name: Target object
        node_tree_name: Name for node tree
        nodes_config: List of node configurations
        
    Returns:
        Geometry nodes setup information
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    # Add geometry nodes modifier
    geo_mod = obj.modifiers.new(name="GeometryNodes", type='NODES')
    
    # Create new node tree
    if not node_tree_name:
        node_tree_name = f"GeometryNodes_{object_name}"
    
    node_tree = bpy.data.node_groups.new(name=node_tree_name, type='GeometryNodeTree')
    geo_mod.node_group = node_tree
    
    nodes = node_tree.nodes
    links = node_tree.links
    
    # Add input and output nodes
    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-200, 0)
    
    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (200, 0)
    
    # Create default socket if needed (Blender 3.0+)
    if hasattr(node_tree, 'inputs'):
        node_tree.inputs.new('NodeSocketGeometry', 'Geometry')
    if hasattr(node_tree, 'outputs'):
        node_tree.outputs.new('NodeSocketGeometry', 'Geometry')
    
    # Connect input to output by default
    links.new(input_node.outputs[0], output_node.inputs[0])
    
    # Add custom nodes if configured
    created_nodes = []
    if nodes_config:
        for config in nodes_config:
            node_type = config.get('type')
            node_pos = config.get('position', [0, 0])
            node_settings = config.get('settings', {})
            
            if node_type:
                new_node = nodes.new(node_type)
                new_node.location = node_pos
                
                # Apply settings
                for key, value in node_settings.items():
                    if hasattr(new_node, key):
                        setattr(new_node, key, value)
                    elif key in new_node.inputs:
                        new_node.inputs[key].default_value = value
                
                created_nodes.append({
                    "type": node_type,
                    "name": new_node.name,
                    "position": list(new_node.location)
                })
    
    return {
        "object": object_name,
        "modifier": geo_mod.name,
        "node_tree": node_tree.name,
        "nodes_created": len(nodes),
        "custom_nodes": created_nodes,
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def create_procedural_geometry(
    object_name: str,
    geometry_type: str = "grid_scatter",
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create complex procedural geometry setups.
    
    Args:
        object_name: Target object
        geometry_type: Type of procedural setup
        parameters: Setup parameters
        
    Returns:
        Procedural geometry configuration
    """
    params = parameters or {}
    
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    # Add geometry nodes
    geo_mod = obj.modifiers.new(name="ProceduralGeometry", type='NODES')
    node_tree = bpy.data.node_groups.new(name=f"Procedural_{geometry_type}", type='GeometryNodeTree')
    geo_mod.node_group = node_tree
    
    nodes = node_tree.nodes
    links = node_tree.links
    
    # Clear default
    nodes.clear()
    
    # Input/Output
    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-400, 0)
    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (400, 0)
    
    # Create sockets
    if hasattr(node_tree, 'inputs'):
        node_tree.inputs.new('NodeSocketGeometry', 'Geometry')
    if hasattr(node_tree, 'outputs'):
        node_tree.outputs.new('NodeSocketGeometry', 'Geometry')
    
    if geometry_type == "grid_scatter":
        # Create grid scatter setup
        distribute = nodes.new('GeometryNodeDistributePointsOnFaces')
        distribute.location = (-200, 0)
        distribute.inputs['Density'].default_value = params.get('density', 50.0)
        
        instance = nodes.new('GeometryNodeInstanceOnPoints')
        instance.location = (0, 0)
        
        cube = nodes.new('GeometryNodeMeshCube')
        cube.location = (-200, -150)
        cube.inputs['Size'].default_value = params.get('instance_size', 0.1)
        
        random_scale = nodes.new('GeometryNodeRandomValue')
        random_scale.location = (-200, -300)
        random_scale.data_type = 'FLOAT_VECTOR'
        random_scale.inputs[0].default_value = params.get('scale_min', [0.5, 0.5, 0.5])
        random_scale.inputs[1].default_value = params.get('scale_max', [1.5, 1.5, 1.5])
        
        links.new(input_node.outputs[0], distribute.inputs['Mesh'])
        links.new(distribute.outputs['Points'], instance.inputs['Points'])
        links.new(cube.outputs['Mesh'], instance.inputs['Instance'])
        links.new(random_scale.outputs['Value'], instance.inputs['Scale'])
        links.new(instance.outputs['Instances'], output_node.inputs[0])
        
    elif geometry_type == "array_circular":
        # Circular array
        curve_circle = nodes.new('GeometryNodeCurveCircle')
        curve_circle.location = (-200, 100)
        curve_circle.inputs['Radius'].default_value = params.get('radius', 2.0)
        curve_circle.inputs['Resolution'].default_value = params.get('count', 12)
        
        curve_to_points = nodes.new('GeometryNodeCurveToPoints')
        curve_to_points.location = (0, 100)
        curve_to_points.mode = 'COUNT'
        curve_to_points.inputs['Count'].default_value = params.get('count', 12)
        
        instance = nodes.new('GeometryNodeInstanceOnPoints')
        instance.location = (200, 0)
        
        links.new(curve_circle.outputs['Curve'], curve_to_points.inputs['Curve'])
        links.new(curve_to_points.outputs['Points'], instance.inputs['Points'])
        links.new(input_node.outputs[0], instance.inputs['Instance'])
        links.new(instance.outputs['Instances'], output_node.inputs[0])
        
    elif geometry_type == "voronoi_fracture":
        # Voronoi fracture effect
        voronoi = nodes.new('ShaderNodeTexVoronoi')
        voronoi.location = (-200, -100)
        voronoi.voronoi_dimensions = '3D'
        voronoi.inputs['Scale'].default_value = params.get('scale', 5.0)
        
        distribute = nodes.new('GeometryNodeDistributePointsOnFaces')
        distribute.location = (0, 0)
        distribute.inputs['Density'].default_value = params.get('density', 100.0)
        
        links.new(input_node.outputs[0], distribute.inputs['Mesh'])
        links.new(distribute.outputs['Points'], output_node.inputs[0])
    
    return {
        "object": object_name,
        "geometry_type": geometry_type,
        "node_tree": node_tree.name,
        "parameters": params,
        "node_count": len(nodes),
        "created_at": datetime.now().isoformat()
    }

# ============================================
# 🎨 UV MAPPING & TEXTURING
# ============================================

@thread_safe
def unwrap_uv(
    object_name: str,
    method: str = "SMART",
    margin: float = 0.001,
    angle_limit: float = 66.0,
    island_margin: float = 0.0,
    area_weight: float = 0.0,
    correct_aspect: bool = True,
    scale_to_bounds: bool = False
) -> Dict[str, Any]:
    """
    UV unwrap object with various methods.
    
    Args:
        object_name: Object to unwrap
        method: SMART, CUBE, SPHERE, CYLINDER, PROJECT, FOLLOW_ACTIVE_QUADS
        margin: Margin between UV islands
        angle_limit: Angle limit for smart project (degrees)
        island_margin: Island margin for smart project
        area_weight: Area weight for smart project
        correct_aspect: Correct aspect ratio
        scale_to_bounds: Scale UVs to bounds
        
    Returns:
        UV unwrap information
    """
    obj = bpy.data.objects.get(object_name)
    if not obj or obj.type != 'MESH':
        raise ValueError(f"Mesh object '{object_name}' not found")
    
    # Ensure object is selected and active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Select all
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Create UV layer if doesn't exist
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    
    # Apply unwrap method
    try:
        if method == "SMART":
            bpy.ops.uv.smart_project(
                angle_limit=math.radians(angle_limit),
                island_margin=island_margin,
                area_weight=area_weight,
                correct_aspect=correct_aspect,
                scale_to_bounds=scale_to_bounds
            )
        elif method == "CUBE":
            bpy.ops.uv.cube_project(
                cube_size=1.0,
                correct_aspect=correct_aspect,
                clip_to_bounds=False,
                scale_to_bounds=scale_to_bounds
            )
        elif method == "SPHERE":
            bpy.ops.uv.sphere_project(
                direction='VIEW_ON_EQUATOR',
                align='POLAR_ZX',
                correct_aspect=correct_aspect,
                clip_to_bounds=False,
                scale_to_bounds=scale_to_bounds
            )
        elif method == "CYLINDER":
            bpy.ops.uv.cylinder_project(
                direction='VIEW_ON_EQUATOR',
                align='POLAR_ZX',
                radius=1.0,
                correct_aspect=correct_aspect,
                clip_to_bounds=False,
                scale_to_bounds=scale_to_bounds
            )
        elif method == "PROJECT":
            bpy.ops.uv.project_from_view(
                camera_bounds=False,
                correct_aspect=correct_aspect,
                scale_to_bounds=scale_to_bounds
            )
        elif method == "FOLLOW_ACTIVE_QUADS":
            bpy.ops.uv.follow_active_quads(mode='LENGTH_AVERAGE')
        else:
            # Default unwrap
            bpy.ops.uv.unwrap(
                method='ANGLE_BASED',
                margin=margin
            )
    finally:
        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Calculate UV statistics
    uv_layer = mesh.uv_layers.active
    uv_data = uv_layer.data
    
    # Get UV bounds
    min_u = min_v = float('inf')
    max_u = max_v = float('-inf')
    
    for uv in uv_data:
        min_u = min(min_u, uv.uv[0])
        max_u = max(max_u, uv.uv[0])
        min_v = min(min_v, uv.uv[1])
        max_v = max(max_v, uv.uv[1])
    
    return {
        "object": object_name,
        "method": method,
        "uv_layer": uv_layer.name,
        "uv_bounds": {
            "min": [min_u, min_v],
            "max": [max_u, max_v],
            "size": [max_u - min_u, max_v - min_v]
        },
        "settings": {
            "margin": margin,
            "angle_limit": angle_limit,
            "correct_aspect": correct_aspect,
            "scale_to_bounds": scale_to_bounds
        },
        "face_count": len(mesh.polygons),
        "uv_count": len(uv_data),
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def add_texture_paint_slots(
    object_name: str,
    texture_types: List[str],
    resolution: int = 2048,
    color_space: str = "sRGB",
    alpha: bool = False,
    float_buffer: bool = False,
    generated_type: str = "BLANK"
) -> Dict[str, Any]:
    """
    Setup texture paint slots for PBR workflow.
    
    Args:
        object_name: Target object
        texture_types: List of texture types (BASE_COLOR, NORMAL, ROUGHNESS, METALLIC, etc.)
        resolution: Texture resolution
        color_space: Color space (sRGB, Non-Color, Linear)
        alpha: Include alpha channel
        float_buffer: Use 32-bit float
        generated_type: BLANK, UV_GRID, COLOR_GRID
        
    Returns:
        Texture slots configuration
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    # Ensure material exists
    if not obj.active_material:
        mat = bpy.data.materials.new(name=f"{object_name}_Material")
        obj.data.materials.append(mat)
    
    mat = obj.active_material
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Find or create Principled BSDF
    bsdf = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            bsdf = node
            break
    
    if not bsdf:
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        output = nodes.get('Material Output')
        if output:
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    created_textures = []
    texture_nodes = {}
    
    for tex_type in texture_types:
        # Create image texture
        img_name = f"{object_name}_{tex_type}"
        img = bpy.data.images.new(
            name=img_name,
            width=resolution,
            height=resolution,
            alpha=alpha,
            float_buffer=float_buffer
        )
        
        # Set color space
        if tex_type in ['NORMAL', 'ROUGHNESS', 'METALLIC', 'DISPLACEMENT']:
            img.colorspace_settings.name = 'Non-Color'
        else:
            img.colorspace_settings.name = color_space
        
        # Generate default content
        if generated_type == "UV_GRID":
            img.generated_type = 'UV_GRID'
        elif generated_type == "COLOR_GRID":
            img.generated_type = 'COLOR_GRID'
        else:
            img.generated_type = 'BLANK'
            if tex_type == 'BASE_COLOR':
                img.generated_color = (0.5, 0.5, 0.5, 1.0)
            elif tex_type == 'ROUGHNESS':
                img.generated_color = (0.5, 0.5, 0.5, 1.0)
            elif tex_type == 'METALLIC':
                img.generated_color = (0.0, 0.0, 0.0, 1.0)
            elif tex_type == 'NORMAL':
                img.generated_color = (0.5, 0.5, 1.0, 1.0)
        
        # Create texture node
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.image = img
        tex_node.location = (-300, len(texture_nodes) * -150)
        texture_nodes[tex_type] = tex_node
        
        # Connect to BSDF
        if tex_type == 'BASE_COLOR' and 'Base Color' in bsdf.inputs:
            links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
        elif tex_type == 'ROUGHNESS' and 'Roughness' in bsdf.inputs:
            links.new(tex_node.outputs['Color'], bsdf.inputs['Roughness'])
        elif tex_type == 'METALLIC' and 'Metallic' in bsdf.inputs:
            links.new(tex_node.outputs['Color'], bsdf.inputs['Metallic'])
        elif tex_type == 'NORMAL' and 'Normal' in bsdf.inputs:
            # Add normal map node
            normal_map = nodes.new('ShaderNodeNormalMap')
            normal_map.location = (-150, len(texture_nodes) * -150)
            links.new(tex_node.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
        
        created_textures.append({
            "name": img_name,
            "type": tex_type,
            "resolution": f"{resolution}x{resolution}",
            "color_space": img.colorspace_settings.name
        })
    
    return {
        "object": object_name,
        "material": mat.name,
        "textures_created": created_textures,
        "texture_count": len(created_textures),
        "resolution": resolution,
        "created_at": datetime.now().isoformat()
    }

# ============================================
# 🚀 BATCH OPERATIONS
# ============================================

@thread_safe
def batch_create_objects(
    operations: List[Dict[str, Any]],
    group_name: Optional[str] = None,
    parent_to: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute multiple object creations efficiently in batch.
    
    Args:
        operations: List of operation dictionaries with 'type' and parameters
        group_name: Optional collection name to group objects
        parent_to: Optional parent object name
        
    Returns:
        Batch operation results
    """
    created_objects = []
    errors = []
    
    # Create collection if specified
    collection = None
    if group_name:
        collection = bpy.data.collections.get(group_name)
        if not collection:
            collection = bpy.data.collections.new(group_name)
            bpy.context.scene.collection.children.link(collection)
    
    # Get parent object if specified
    parent_obj = None
    if parent_to:
        parent_obj = bpy.data.objects.get(parent_to)
    
    # Process operations
    for i, op in enumerate(operations):
        try:
            op_type = op.get('type')
            params = op.get('parameters', {})
            
            # Execute based on operation type
            if op_type == 'mesh':
                result = create_mesh_object(**params)
            elif op_type == 'curve':
                result = create_curve_object(**params)
            elif op_type == 'text':
                result = create_text_object(**params)
            elif op_type == 'light':
                result = create_light(**params)
            elif op_type == 'camera':
                result = create_camera(**params)
            elif op_type == 'empty':
                bpy.ops.object.empty_add(
                    type=params.get('display_type', 'PLAIN_AXES'),
                    location=params.get('location', [0, 0, 0])
                )
                result = {"name": bpy.context.active_object.name, "type": "empty"}
            else:
                raise ValueError(f"Unknown operation type: {op_type}")
            
            # Handle collection and parenting
            if result and 'name' in result:
                obj = bpy.data.objects.get(result['name'])
                
                if obj and collection:
                    # Move to collection
                    for coll in obj.users_collection:
                        coll.objects.unlink(obj)
                    collection.objects.link(obj)
                
                if obj and parent_obj:
                    obj.parent = parent_obj
                    obj.matrix_parent_inverse = parent_obj.matrix_world.inverted()
            
            created_objects.append({
                "index": i,
                "type": op_type,
                "name": result.get('name'),
                "success": True
            })
            
        except Exception as e:
            errors.append({
                "index": i,
                "type": op.get('type'),
                "error": str(e)
            })
    
    return {
        "total_operations": len(operations),
        "successful": len(created_objects),
        "failed": len(errors),
        "created_objects": created_objects,
        "errors": errors,
        "collection": group_name,
        "parent": parent_to,
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def batch_transform(
    objects: List[str],
    transformations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply transformations to multiple objects at once.
    
    Args:
        objects: List of object names
        transformations: Dictionary with location, rotation, scale operations
        
    Returns:
        Batch transformation results
    """
    results = []
    
    location = transformations.get('location')
    rotation = transformations.get('rotation')
    scale = transformations.get('scale')
    delta_mode = transformations.get('delta', False)
    randomize = transformations.get('randomize', {})
    
    for obj_name in objects:
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            results.append({"object": obj_name, "success": False, "error": "Not found"})
            continue
        
        try:
            # Apply location
            if location:
                loc = location.copy()
                if randomize.get('location'):
                    rand_range = randomize['location']
                    loc = [loc[i] + random.uniform(-rand_range, rand_range) for i in range(3)]
                
                if delta_mode:
                    obj.location = [obj.location[i] + loc[i] for i in range(3)]
                else:
                    obj.location = loc
            
            # Apply rotation
            if rotation:
                rot = rotation.copy()
                if randomize.get('rotation'):
                    rand_range = randomize['rotation']
                    rot = [rot[i] + random.uniform(-rand_range, rand_range) for i in range(3)]
                
                rot_rad = [math.radians(r) for r in rot]
                if delta_mode:
                    obj.rotation_euler = [obj.rotation_euler[i] + rot_rad[i] for i in range(3)]
                else:
                    obj.rotation_euler = rot_rad
            
            # Apply scale
            if scale:
                scl = scale.copy() if isinstance(scale, list) else [scale, scale, scale]
                if randomize.get('scale'):
                    rand_range = randomize['scale']
                    scl = [scl[i] * random.uniform(1-rand_range, 1+rand_range) for i in range(3)]
                
                if delta_mode:
                    obj.scale = [obj.scale[i] * scl[i] for i in range(3)]
                else:
                    obj.scale = scl
            
            results.append({
                "object": obj_name,
                "success": True,
                "new_location": list(obj.location),
                "new_rotation": [math.degrees(r) for r in obj.rotation_euler],
                "new_scale": list(obj.scale)
            })
            
        except Exception as e:
            results.append({"object": obj_name, "success": False, "error": str(e)})
    
    return {
        "objects_processed": len(objects),
        "successful": len([r for r in results if r["success"]]),
        "delta_mode": delta_mode,
        "randomization": bool(randomize),
        "results": results,
        "processed_at": datetime.now().isoformat()
    }

# ============================================
# ⚙️ SIMULATION SYSTEMS
# ============================================

@thread_safe
def setup_rigid_body(
    object_name: str,
    body_type: str = "ACTIVE",
    shape: str = "CONVEX_HULL",
    mass: float = 1.0,
    friction: float = 0.5,
    bounciness: float = 0.0,
    use_margin: bool = False,
    collision_margin: float = 0.04,
    linear_damping: float = 0.04,
    angular_damping: float = 0.1,
    collision_collections: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Setup rigid body physics simulation.
    
    Args:
        object_name: Object to make rigid body
        body_type: ACTIVE or PASSIVE
        shape: BOX, SPHERE, CAPSULE, CYLINDER, CONE, CONVEX_HULL, MESH, COMPOUND
        mass: Object mass
        friction: Surface friction
        bounciness: Restitution/bounciness
        use_margin: Use collision margin
        collision_margin: Margin size
        linear_damping: Linear motion damping
        angular_damping: Angular motion damping
        collision_collections: Collision layers (1-20)
        
    Returns:
        Rigid body configuration
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    # Enable rigid body
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    
    rb = obj.rigid_body
    
    # Configure rigid body
    rb.type = body_type
    rb.collision_shape = shape
    rb.mass = mass
    rb.friction = friction
    rb.restitution = bounciness
    rb.use_margin = use_margin
    rb.collision_margin = collision_margin
    rb.linear_damping = linear_damping
    rb.angular_damping = angular_damping
    
    # Set collision collections
    if collision_collections:
        for i in range(20):
            rb.collision_collections[i] = (i + 1) in collision_collections
    
    # Calculate physics properties
    volume = 0
    if hasattr(obj.data, 'vertices'):
        # Estimate volume from bounding box
        dims = obj.dimensions
        volume = dims[0] * dims[1] * dims[2]
    
    density = mass / volume if volume > 0 else 1.0
    
    return {
        "object": object_name,
        "rigid_body": {
            "type": body_type,
            "shape": shape,
            "mass": mass,
            "friction": friction,
            "bounciness": bounciness,
            "linear_damping": linear_damping,
            "angular_damping": angular_damping
        },
        "physics": {
            "estimated_volume": volume,
            "density": density,
            "collision_margin": collision_margin if use_margin else None
        },
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def add_cloth_simulation(
    object_name: str,
    quality: int = 5,
    mass: float = 0.3,
    tension_stiffness: float = 15.0,
    compression_stiffness: float = 15.0,
    shear_stiffness: float = 5.0,
    bending_stiffness: float = 0.5,
    damping_tension: float = 5.0,
    damping_compression: float = 5.0,
    damping_shear: float = 5.0,
    damping_bending: float = 0.5,
    air_damping: float = 1.0,
    use_collision: bool = True,
    collision_distance: float = 0.015,
    use_self_collision: bool = False,
    self_collision_distance: float = 0.015,
    vertex_group_mass: Optional[str] = None,
    pin_vertex_group: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add realistic cloth simulation.
    
    Args:
        object_name: Object to make cloth
        quality: Simulation quality steps
        mass: Cloth mass (kg/m²)
        tension_stiffness: Tension spring stiffness
        compression_stiffness: Compression spring stiffness
        shear_stiffness: Shear spring stiffness
        bending_stiffness: Bending spring stiffness
        damping_tension: Tension damping
        damping_compression: Compression damping
        damping_shear: Shear damping
        damping_bending: Bending damping
        air_damping: Air resistance
        use_collision: Enable collision
        collision_distance: Collision distance
        use_self_collision: Enable self collision
        self_collision_distance: Self collision distance
        vertex_group_mass: Vertex group for mass variation
        pin_vertex_group: Vertex group for pinning
        
    Returns:
        Cloth simulation configuration
    """
    obj = bpy.data.objects.get(object_name)
    if not obj or obj.type != 'MESH':
        raise ValueError(f"Mesh object '{object_name}' not found")
    
    # Add cloth modifier
    cloth_mod = obj.modifiers.new(name="Cloth", type='CLOTH')
    cloth = cloth_mod.settings
    
    # Quality settings
    cloth.quality = quality
    
    # Physical properties
    cloth.mass = mass
    cloth.air_damping = air_damping
    
    # Stiffness settings
    cloth.tension_stiffness = tension_stiffness
    cloth.compression_stiffness = compression_stiffness
    cloth.shear_stiffness = shear_stiffness
    cloth.bending_stiffness = bending_stiffness
    
    # Damping settings
    cloth.tension_damping = damping_tension
    cloth.compression_damping = damping_compression
    cloth.shear_damping = damping_shear
    cloth.bending_damping = damping_bending
    
    # Collision settings
    if use_collision:
        cloth.collision_settings.use_collision = True
        cloth.collision_settings.distance_min = collision_distance
        cloth.collision_settings.use_self_collision = use_self_collision
        if use_self_collision:
            cloth.collision_settings.self_distance_min = self_collision_distance
    
    # Vertex groups
    if pin_vertex_group:
        cloth.vertex_group_mass = pin_vertex_group
    if vertex_group_mass:
        cloth.vertex_group_mass = vertex_group_mass
    
    return {
        "object": object_name,
        "modifier": cloth_mod.name,
        "settings": {
            "quality": quality,
            "mass": mass,
            "air_damping": air_damping
        },
        "stiffness": {
            "tension": tension_stiffness,
            "compression": compression_stiffness,
            "shear": shear_stiffness,
            "bending": bending_stiffness
        },
        "damping": {
            "tension": damping_tension,
            "compression": damping_compression,
            "shear": damping_shear,
            "bending": damping_bending
        },
        "collision": {
            "enabled": use_collision,
            "distance": collision_distance,
            "self_collision": use_self_collision
        },
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def setup_fluid_simulation(
    domain_name: str,
    resolution: int = 64,
    simulation_type: str = "LIQUID",
    time_scale: float = 1.0,
    use_adaptive_domain: bool = True,
    use_collision_objects: bool = True,
    viscosity: float = 0.0,
    surface_tension: float = 0.0,
    use_foam: bool = False,
    use_spray: bool = False,
    use_bubbles: bool = False
) -> Dict[str, Any]:
    """
    Setup complete fluid simulation domain.
    
    Args:
        domain_name: Domain object name
        resolution: Simulation resolution
        simulation_type: LIQUID or GAS
        time_scale: Time scale factor
        use_adaptive_domain: Adaptive domain sizing
        use_collision_objects: Enable collisions
        viscosity: Fluid viscosity
        surface_tension: Surface tension
        use_foam: Generate foam particles
        use_spray: Generate spray particles
        use_bubbles: Generate bubble particles
        
    Returns:
        Fluid simulation configuration
    """
    obj = bpy.data.objects.get(domain_name)
    if not obj:
        raise ValueError(f"Object '{domain_name}' not found")
    
    # Add fluid modifier
    fluid_mod = obj.modifiers.new(name="Fluid", type='FLUID')
    fluid_mod.fluid_type = 'DOMAIN'
    
    domain = fluid_mod.domain_settings
    
    # Basic settings
    domain.domain_type = simulation_type
    domain.resolution_max = resolution
    domain.time_scale = time_scale
    domain.use_adaptive_domain = use_adaptive_domain
    domain.use_collision_objects = use_collision_objects
    
    # Liquid specific settings
    if simulation_type == "LIQUID":
        domain.use_flip_particles = True
        domain.flip_ratio = 0.97
        domain.particle_radius = 1.5
        
        # Viscosity
        if viscosity > 0:
            domain.use_viscosity = True
            domain.viscosity_value = viscosity
        
        # Surface tension
        if surface_tension > 0:
            domain.use_surface_tension = True
            domain.surface_tension = surface_tension
        
        # Secondary particles
        if use_foam:
            domain.use_foam_particles = True
            domain.foam_lifetime = 2.0
        if use_spray:
            domain.use_spray_particles = True
            domain.spray_lifetime = 1.5
        if use_bubbles:
            domain.use_bubble_particles = True
            domain.bubble_lifetime = 1.0
    
    # Gas specific settings
    elif simulation_type == "GAS":
        domain.use_noise = True
        domain.noise_scale = 2.0
        domain.noise_strength = 1.0
    
    # Cache settings
    domain.cache_type = 'MODULAR'
    domain.cache_directory = f"//cache_fluid_{domain_name}"
    
    return {
        "domain": domain_name,
        "type": simulation_type,
        "resolution": resolution,
        "settings": {
            "time_scale": time_scale,
            "adaptive_domain": use_adaptive_domain,
            "collision_objects": use_collision_objects,
            "viscosity": viscosity,
            "surface_tension": surface_tension
        },
        "particles": {
            "foam": use_foam,
            "spray": use_spray,
            "bubbles": use_bubbles
        },
        "cache_directory": domain.cache_directory,
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def add_fluid_flow(
    object_name: str,
    flow_type: str = "INFLOW",
    flow_behavior: str = "INFLOW",
    flow_rate: float = 1.0,
    temperature: float = 1.0,
    density: float = 1.0,
    fuel_amount: float = 1.0,
    smoke_color: Optional[List[float]] = None,
    use_texture: bool = False,
    texture_size: float = 1.0,
    texture_offset: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Add fluid flow source/obstacle.
    
    Args:
        object_name: Flow object
        flow_type: INFLOW, OUTFLOW, SMOKE, FIRE, LIQUID
        flow_behavior: INFLOW, OUTFLOW, GEOMETRY
        flow_rate: Flow emission rate
        temperature: Temperature differential
        density: Smoke density
        fuel_amount: Fuel for fire
        smoke_color: Smoke color
        use_texture: Use texture for flow
        texture_size: Texture scale
        texture_offset: Texture offset
        
    Returns:
        Flow configuration
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    smoke_color = smoke_color or [0.7, 0.7, 0.7]
    texture_offset = texture_offset or [0, 0, 0]
    
    # Add fluid modifier
    fluid_mod = obj.modifiers.new(name="Fluid", type='FLUID')
    fluid_mod.fluid_type = 'FLOW'
    
    flow = fluid_mod.flow_settings
    
    # Configure flow
    flow.flow_type = flow_type
    flow.flow_behavior = flow_behavior
    flow.subframes = 1
    
    # Set flow properties based on type
    if flow_type in ['SMOKE', 'FIRE', 'BOTH']:
        flow.smoke_color = smoke_color
        flow.temperature = temperature
        flow.density = density
        if flow_type in ['FIRE', 'BOTH']:
            flow.fuel_amount = fuel_amount
    
    elif flow_type == 'LIQUID':
        flow.use_initial_velocity = True
        flow.velocity_factor = flow_rate
    
    # Texture settings
    if use_texture:
        flow.use_texture = True
        flow.texture_size = texture_size
        flow.texture_offset = texture_offset
    
    return {
        "object": object_name,
        "flow_type": flow_type,
        "flow_behavior": flow_behavior,
        "properties": {
            "temperature": temperature,
            "density": density,
            "fuel_amount": fuel_amount if flow_type in ['FIRE', 'BOTH'] else None,
            "smoke_color": smoke_color
        },
        "texture": {
            "enabled": use_texture,
            "size": texture_size,
            "offset": texture_offset
        },
        "created_at": datetime.now().isoformat()
    }

# ============================================
# 🎭 ADVANCED NODE EDITOR
# ============================================

@thread_safe
def create_shader_node_tree(
    material_name: str,
    nodes: List[Dict[str, Any]],
    connections: List[Tuple[str, str, str, str]]
) -> Dict[str, Any]:
    """
    Create complex shader networks programmatically.
    
    Args:
        material_name: Material to modify
        nodes: List of node definitions with type, name, location, settings
        connections: List of (from_node, from_socket, to_node, to_socket)
        
    Returns:
        Shader network configuration
    """
    mat = bpy.data.materials.get(material_name)
    if not mat:
        mat = bpy.data.materials.new(name=material_name)
    
    mat.use_nodes = True
    node_tree = mat.node_tree
    tree_nodes = node_tree.nodes
    links = node_tree.links
    
    # Clear existing nodes if requested
    created_nodes = {}
    
    # Create nodes
    for node_def in nodes:
        node_type = node_def.get('type')
        node_name = node_def.get('name', node_type)
        location = node_def.get('location', [0, 0])
        settings = node_def.get('settings', {})
        
        # Create node
        new_node = tree_nodes.new(node_type)
        new_node.name = node_name
        new_node.location = location
        
        # Apply settings
        for key, value in settings.items():
            if hasattr(new_node, key):
                setattr(new_node, key, value)
            elif key in new_node.inputs:
                new_node.inputs[key].default_value = value
        
        created_nodes[node_name] = new_node
    
    # Create connections
    for from_node, from_socket, to_node, to_socket in connections:
        if from_node in created_nodes and to_node in created_nodes:
            from_n = created_nodes[from_node]
            to_n = created_nodes[to_node]
            
            # Find sockets
            from_s = None
            to_s = None
            
            if from_socket in from_n.outputs:
                from_s = from_n.outputs[from_socket]
            elif from_socket.isdigit():
                from_s = from_n.outputs[int(from_socket)]
            
            if to_socket in to_n.inputs:
                to_s = to_n.inputs[to_socket]
            elif to_socket.isdigit():
                to_s = to_n.inputs[int(to_socket)]
            
            if from_s and to_s:
                links.new(from_s, to_s)
    
    return {
        "material": material_name,
        "nodes_created": len(created_nodes),
        "connections_made": len(connections),
        "node_tree": node_tree.name,
        "nodes": list(created_nodes.keys()),
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def create_procedural_material(
    name: str,
    material_type: str = "wood",
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create procedural materials with customizable parameters.
    
    Args:
        name: Material name
        material_type: wood, marble, concrete, metal, fabric, glass, etc.
        parameters: Type-specific parameters
        
    Returns:
        Procedural material configuration
    """
    params = parameters or {}
    
    # Create material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Add output node
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (800, 0)
    
    # Create material based on type
    if material_type == "wood":
        # Wood procedural material
        coord = nodes.new('ShaderNodeTexCoord')
        coord.location = (-800, 0)
        
        mapping = nodes.new('ShaderNodeMapping')
        mapping.location = (-600, 0)
        mapping.inputs['Scale'].default_value = params.get('scale', [1, 1, 20])
        
        wave1 = nodes.new('ShaderNodeTexWave')
        wave1.location = (-400, 100)
        wave1.wave_type = 'RINGS'
        wave1.inputs['Scale'].default_value = params.get('ring_scale', 5.0)
        wave1.inputs['Distortion'].default_value = params.get('distortion', 2.0)
        
        noise = nodes.new('ShaderNodeTexNoise')
        noise.location = (-400, -100)
        noise.inputs['Scale'].default_value = params.get('noise_scale', 50.0)
        noise.inputs['Detail'].default_value = params.get('detail', 5.0)
        
        mix = nodes.new('ShaderNodeMixRGB')
        mix.location = (-200, 0)
        mix.blend_type = 'MIX'
        mix.inputs['Fac'].default_value = params.get('mix_factor', 0.2)
        
        colorramp = nodes.new('ShaderNodeValToRGB')
        colorramp.location = (0, 0)
        colorramp.color_ramp.elements[0].color = params.get('color1', [0.1, 0.05, 0.01, 1])
        colorramp.color_ramp.elements[1].color = params.get('color2', [0.3, 0.15, 0.05, 1])
        
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (400, 0)
        bsdf.inputs['Roughness'].default_value = params.get('roughness', 0.8)
        
        # Connect nodes
        links.new(coord.outputs['Object'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], wave1.inputs['Vector'])
        links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
        links.new(wave1.outputs['Fac'], mix.inputs['Color1'])
        links.new(noise.outputs['Fac'], mix.inputs['Color2'])
        links.new(mix.outputs['Color'], colorramp.inputs['Fac'])
        links.new(colorramp.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
    elif material_type == "marble":
        # Marble procedural material
        coord = nodes.new('ShaderNodeTexCoord')
        coord.location = (-800, 0)
        
        noise1 = nodes.new('ShaderNodeTexNoise')
        noise1.location = (-400, 100)
        noise1.inputs['Scale'].default_value = params.get('scale', 2.0)
        noise1.inputs['Detail'].default_value = params.get('detail', 5.0)
        noise1.inputs['Distortion'].default_value = params.get('distortion', 2.0)
        
        voronoi = nodes.new('ShaderNodeTexVoronoi')
        voronoi.location = (-400, -100)
        voronoi.voronoi_dimensions = '3D'
        voronoi.feature = 'F1'
        voronoi.inputs['Scale'].default_value = params.get('vein_scale', 5.0)
        
        mix = nodes.new('ShaderNodeMixRGB')
        mix.location = (-200, 0)
        mix.blend_type = 'LINEAR_LIGHT'
        mix.inputs['Fac'].default_value = params.get('mix', 0.5)
        
        colorramp = nodes.new('ShaderNodeValToRGB')
        colorramp.location = (0, 0)
        colorramp.color_ramp.elements[0].color = params.get('color1', [0.8, 0.8, 0.8, 1])
        colorramp.color_ramp.elements[1].color = params.get('color2', [0.2, 0.2, 0.25, 1])
        
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (400, 0)
        bsdf.inputs['Roughness'].default_value = params.get('roughness', 0.1)
        bsdf.inputs['IOR'].default_value = params.get('ior', 1.46)
        
        # Connect nodes
        links.new(coord.outputs['Object'], noise1.inputs['Vector'])
        links.new(coord.outputs['Object'], voronoi.inputs['Vector'])
        links.new(noise1.outputs['Fac'], mix.inputs['Color1'])
        links.new(voronoi.outputs['Distance'], mix.inputs['Color2'])
        links.new(mix.outputs['Color'], colorramp.inputs['Fac'])
        links.new(colorramp.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
    elif material_type == "concrete":
        # Concrete procedural material
        coord = nodes.new('ShaderNodeTexCoord')
        coord.location = (-600, 0)
        
        noise = nodes.new('ShaderNodeTexNoise')
        noise.location = (-400, 0)
        noise.inputs['Scale'].default_value = params.get('scale', 10.0)
        noise.inputs['Detail'].default_value = params.get('detail', 15.0)
        noise.inputs['Roughness'].default_value = params.get('roughness', 0.9)
        
        musgrave = nodes.new('ShaderNodeTexMusgrave')
        musgrave.location = (-200, -100)
        musgrave.musgrave_type = 'FBM'
        musgrave.inputs['Scale'].default_value = params.get('texture_scale', 50.0)
        
        bump = nodes.new('ShaderNodeBump')
        bump.location = (0, -100)
        bump.inputs['Strength'].default_value = params.get('bump_strength', 0.1)
        
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (200, 0)
        bsdf.inputs['Base Color'].default_value = params.get('color', [0.5, 0.5, 0.5, 1])
        bsdf.inputs['Roughness'].default_value = params.get('surface_roughness', 0.9)
        
        # Connect nodes
        links.new(coord.outputs['Object'], noise.inputs['Vector'])
        links.new(coord.outputs['Object'], musgrave.inputs['Vector'])
        links.new(musgrave.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
        links.new(noise.outputs['Fac'], bsdf.inputs['Roughness'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    else:
        # Default principled material
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        bsdf.inputs['Base Color'].default_value = params.get('color', [0.5, 0.5, 0.5, 1])
        bsdf.inputs['Roughness'].default_value = params.get('roughness', 0.5)
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return {
        "material": mat.name,
        "type": material_type,
        "parameters": params,
        "node_count": len(nodes),
        "created_at": datetime.now().isoformat()
    }

# ============================================
# 🏗️ TEMPLATE SYSTEM
# ============================================

@thread_safe
def create_from_template(
    template_name: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create complex objects from predefined templates.
    
    Args:
        template_name: Template type (character, vehicle, architecture, tree, etc.)
        parameters: Template-specific parameters
        
    Returns:
        Template creation results
    """
    params = parameters or {}
    results = {}
    
    if template_name == "character":
        # Create character with armature
        height = params.get('height', 1.8)
        
        # Create body parts
        # Head
        head = create_mesh_object(
            primitive_type="sphere",
            size=height * 0.15,
            location=[0, 0, height * 0.9],
            name="Character_Head"
        )
        
        # Body
        body = create_mesh_object(
            primitive_type="cube",
            size=height * 0.3,
            location=[0, 0, height * 0.5],
            scale=[1, 0.5, 1.5],
            name="Character_Body"
        )
        
        # Arms
        arm_l = create_mesh_object(
            primitive_type="cylinder",
            size=height * 0.3,
            location=[-height * 0.2, 0, height * 0.6],
            rotation=[0, 90, 0],
            scale=[0.3, 0.3, 1],
            name="Character_Arm_L"
        )
        
        arm_r = create_mesh_object(
            primitive_type="cylinder",
            size=height * 0.3,
            location=[height * 0.2, 0, height * 0.6],
            rotation=[0, 90, 0],
            scale=[0.3, 0.3, 1],
            name="Character_Arm_R"
        )
        
        # Legs
        leg_l = create_mesh_object(
            primitive_type="cylinder",
            size=height * 0.4,
            location=[-height * 0.08, 0, height * 0.2],
            scale=[0.3, 0.3, 1],
            name="Character_Leg_L"
        )
        
        leg_r = create_mesh_object(
            primitive_type="cylinder",
            size=height * 0.4,
            location=[height * 0.08, 0, height * 0.2],
            scale=[0.3, 0.3, 1],
            name="Character_Leg_R"
        )
        
        # Create armature if requested
        if params.get('add_armature', True):
            bpy.ops.object.armature_add(location=[0, 0, 0])
            armature = bpy.context.active_object
            armature.name = "Character_Armature"
            
            # Setup basic bones in edit mode
            bpy.ops.object.mode_set(mode='EDIT')
            bones = armature.data.edit_bones
            
            # Create spine
            spine = bones.new('Spine')
            spine.head = [0, 0, height * 0.3]
            spine.tail = [0, 0, height * 0.7]
            
            # Create head bone
            head_bone = bones.new('Head')
            head_bone.head = spine.tail
            head_bone.tail = [0, 0, height]
            head_bone.parent = spine
            
            bpy.ops.object.mode_set(mode='OBJECT')
        
        results = {
            "template": "character",
            "objects_created": 6,
            "height": height,
            "armature": params.get('add_armature', True)
        }
        
    elif template_name == "vehicle":
        # Create basic vehicle
        length = params.get('length', 4)
        width = params.get('width', 2)
        height = params.get('height', 1.5)
        
        # Body
        body = create_mesh_object(
            primitive_type="cube",
            size=1,
            scale=[length, width, height],
            location=[0, 0, height * 0.7],
            name="Vehicle_Body"
        )
        
        # Wheels
        wheel_positions = [
            [-length * 0.35, -width * 0.5, height * 0.3],
            [length * 0.35, -width * 0.5, height * 0.3],
            [-length * 0.35, width * 0.5, height * 0.3],
            [length * 0.35, width * 0.5, height * 0.3]
        ]
        
        for i, pos in enumerate(wheel_positions):
            wheel = create_mesh_object(
                primitive_type="cylinder",
                size=height * 0.4,
                location=pos,
                rotation=[90, 0, 0],
                name=f"Vehicle_Wheel_{i+1}"
            )
        
        results = {
            "template": "vehicle",
            "objects_created": 5,
            "dimensions": [length, width, height]
        }
        
    elif template_name == "building":
        # Create parametric building
        floors = params.get('floors', 5)
        width = params.get('width', 10)
        depth = params.get('depth', 10)
        floor_height = params.get('floor_height', 3)
        
        # Base structure
        building = create_mesh_object(
            primitive_type="cube",
            size=1,
            scale=[width, depth, floors * floor_height],
            location=[0, 0, floors * floor_height * 0.5],
            name="Building_Structure"
        )
        
        # Add windows if requested
        if params.get('add_windows', True):
            window_mod = add_modifier(
                object_name="Building_Structure",
                modifier_type="ARRAY",
                settings={
                    'count': floors,
                    'use_relative_offset': True,
                    'offset_z': 1.0 / floors
                }
            )
        
        results = {
            "template": "building",
            "floors": floors,
            "dimensions": [width, depth, floors * floor_height]
        }
        
    elif template_name == "tree":
        # Create procedural tree
        trunk_height = params.get('trunk_height', 4)
        crown_size = params.get('crown_size', 3)
        
        # Trunk
        trunk = create_mesh_object(
            primitive_type="cylinder",
            size=trunk_height,
            location=[0, 0, trunk_height * 0.5],
            scale=[0.3, 0.3, 1],
            name="Tree_Trunk"
        )
        
        # Crown
        crown = create_mesh_object(
            primitive_type="ico_sphere",
            size=crown_size,
            location=[0, 0, trunk_height + crown_size * 0.4],
            ico_subdivisions=2,
            name="Tree_Crown"
        )
        
        # Add materials
        if params.get('add_materials', True):
            # Trunk material
            trunk_mat = create_material(
                name="Tree_Bark",
                base_color=[0.3, 0.2, 0.1, 1],
                roughness=0.9
            )
            assign_material("Tree_Trunk", "Tree_Bark")
            
            # Crown material
            crown_mat = create_material(
                name="Tree_Leaves",
                base_color=[0.1, 0.5, 0.1, 1],
                roughness=0.8
            )
            assign_material("Tree_Crown", "Tree_Leaves")
        
        results = {
            "template": "tree",
            "trunk_height": trunk_height,
            "crown_size": crown_size,
            "materials_added": params.get('add_materials', True)
        }
    
    else:
        raise ValueError(f"Unknown template: {template_name}")
    
    results["created_at"] = datetime.now().isoformat()
    return results

# ============================================
# ✏️ GREASE PENCIL
# ============================================

@thread_safe
def create_grease_pencil_drawing(
    name: str,
    strokes: List[Dict[str, Any]],
    materials: Optional[List[Dict[str, Any]]] = None,
    frame: int = 1
) -> Dict[str, Any]:
    """
    Create 2D drawings in 3D space using Grease Pencil.
    
    Args:
        name: Grease pencil object name
        strokes: List of stroke definitions
        materials: List of material definitions
        frame: Frame to draw on
        
    Returns:
        Grease pencil configuration
    """
    # Create grease pencil data
    gp_data = bpy.data.grease_pencils.new(name)
    
    # Create grease pencil object
    gp_obj = bpy.data.objects.new(name, gp_data)
    bpy.context.collection.objects.link(gp_obj)
    
    # Create layer
    gp_layer = gp_data.layers.new('Drawing_Layer')
    
    # Create frame
    gp_frame = gp_layer.frames.new(frame)
    
    # Create materials
    created_materials = []
    if materials:
        for mat_def in materials:
            mat = bpy.data.materials.new(mat_def.get('name', 'GP_Material'))
            mat.grease_pencil.show_stroke = mat_def.get('show_stroke', True)
            mat.grease_pencil.show_fill = mat_def.get('show_fill', False)
            mat.grease_pencil.color = mat_def.get('color', [0, 0, 0, 1])
            gp_data.materials.append(mat)
            created_materials.append(mat.name)
    
    # Create strokes
    created_strokes = []
    for stroke_def in strokes:
        stroke = gp_frame.strokes.new()
        
        # Set stroke properties
        stroke.line_width = stroke_def.get('line_width', 10)
        stroke.material_index = stroke_def.get('material_index', 0)
        stroke.use_cyclic = stroke_def.get('cyclic', False)
        
        # Add points
        points = stroke_def.get('points', [[0, 0, 0]])
        stroke.points.add(len(points))
        
        for i, point in enumerate(points):
            stroke.points[i].co = point
            stroke.points[i].pressure = stroke_def.get('pressure', 1.0)
            stroke.points[i].strength = stroke_def.get('strength', 1.0)
        
        created_strokes.append({
            "point_count": len(points),
            "line_width": stroke.line_width,
            "cyclic": stroke.use_cyclic
        })
    
    return {
        "name": gp_obj.name,
        "layers": 1,
        "frame": frame,
        "strokes_created": len(created_strokes),
        "materials_created": len(created_materials),
        "stroke_details": created_strokes,
        "created_at": datetime.now().isoformat()
    }

# ============================================
# ⚡ PERFORMANCE & OPTIMIZATION
# ============================================

@lru_cache(maxsize=Config.CACHE_SIZE)
def get_cached_object_info(object_name: str) -> Optional[Dict[str, Any]]:
    """
    Get cached object information for performance.
    
    Args:
        object_name: Object to query
        
    Returns:
        Cached object data or None
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        return None
    
    return {
        "name": obj.name,
        "type": obj.type,
        "location": list(obj.location),
        "rotation": [math.degrees(r) for r in obj.rotation_euler],
        "scale": list(obj.scale),
        "visible": obj.visible_get(),
        "selected": obj.select_get()
    }

def clear_cache():
    """Clear all caches"""
    get_cached_object_info.cache_clear()
    logger.info("Cache cleared")

@thread_safe
def optimize_scene(
    merge_objects: bool = False,
    remove_doubles: bool = True,
    decimate_ratio: float = 1.0,
    remove_unused_data: bool = True
) -> Dict[str, Any]:
    """
    Optimize scene for better performance.
    
    Args:
        merge_objects: Merge selected objects
        remove_doubles: Remove duplicate vertices
        decimate_ratio: Decimation ratio (1.0 = no decimation)
        remove_unused_data: Remove unused data blocks
        
    Returns:
        Optimization results
    """
    results = {
        "objects_processed": 0,
        "vertices_removed": 0,
        "unused_data_removed": 0
    }
    
    # Process meshes
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh = obj.data
            
            # Remove doubles
            if remove_doubles:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                
                before = len(mesh.vertices)
                bpy.ops.mesh.remove_doubles(threshold=0.0001)
                after = len(mesh.vertices)
                
                results["vertices_removed"] += before - after
                bpy.ops.object.mode_set(mode='OBJECT')
            
            # Add decimate if needed
            if decimate_ratio < 1.0:
                decimate = obj.modifiers.new('Decimate', 'DECIMATE')
                decimate.ratio = decimate_ratio
            
            results["objects_processed"] += 1
    
    # Remove unused data
    if remove_unused_data:
        # Remove unused meshes
        for mesh in bpy.data.meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
                results["unused_data_removed"] += 1
        
        # Remove unused materials
        for mat in bpy.data.materials:
            if mat.users == 0:
                bpy.data.materials.remove(mat)
                results["unused_data_removed"] += 1
        
        # Remove unused textures
        for img in bpy.data.images:
            if img.users == 0:
                bpy.data.images.remove(img)
                results["unused_data_removed"] += 1
    
    results["optimized_at"] = datetime.now().isoformat()
    return results

# ============================================
# 🎯 PRESETS & QUICK ACTIONS
# ============================================

@thread_safe
def quick_scene_setup(
    preset: str = "product_viz"
) -> Dict[str, Any]:
    """
    Quick scene setups with optimal settings.
    
    Args:
        preset: Preset name (product_viz, architecture, character_animation, etc.)
        
    Returns:
        Scene setup results
    """
    results = {}
    
    presets = {
        "product_viz": {
            "lighting": "three_point",
            "camera_settings": {
                "focal_length": 85,
                "location": [3, -3, 2],
                "rotation": [65, 0, 45]
            },
            "render_settings": {
                "engine": "CYCLES",
                "samples": 256,
                "resolution_x": 1920,
                "resolution_y": 1080,
                "transparent": True
            },
            "world_settings": {
                "use_hdri": False,
                "background_color": [0.05, 0.05, 0.05],
                "ambient_occlusion": True
            }
        },
        "architecture": {
            "lighting": "sun_sky",
            "camera_settings": {
                "focal_length": 24,
                "location": [15, -15, 10],
                "rotation": [70, 0, 45]
            },
            "render_settings": {
                "engine": "CYCLES",
                "samples": 512,
                "resolution_x": 3840,
                "resolution_y": 2160,
                "use_denoising": True
            },
            "world_settings": {
                "use_hdri": True,
                "sun_rotation": 45
            }
        },
        "character_animation": {
            "lighting": "studio",
            "camera_settings": {
                "focal_length": 50,
                "location": [5, -5, 2],
                "rotation": [80, 0, 45]
            },
            "render_settings": {
                "engine": "EEVEE",
                "samples": 64,
                "resolution_x": 1920,
                "resolution_y": 1080,
                "use_motion_blur": True
            },
            "world_settings": {
                "use_hdri": False,
                "background_color": [0.1, 0.1, 0.1]
            }
        },
        "motion_graphics": {
            "lighting": "flat",
            "camera_settings": {
                "focal_length": 35,
                "location": [0, -10, 0],
                "rotation": [90, 0, 0]
            },
            "render_settings": {
                "engine": "EEVEE",
                "samples": 32,
                "resolution_x": 1920,
                "resolution_y": 1080,
                "use_bloom": True
            },
            "world_settings": {
                "use_hdri": False,
                "background_color": [1, 1, 1]
            }
        }
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}")
    
    settings = presets[preset]
    
    # Apply camera settings
    cam = create_camera(
        name=f"{preset}_Camera",
        **settings["camera_settings"]
    )
    bpy.context.scene.camera = bpy.data.objects.get(cam["name"])
    results["camera"] = cam["name"]
    
    # Apply render settings
    render = configure_render_settings(**settings["render_settings"])
    results["render"] = render
    
    # Setup lighting
    lighting_preset = settings["lighting"]
    if lighting_preset == "three_point":
        # Key light
        key = create_light(
            light_type="AREA",
            name="Key_Light",
            location=[5, -5, 5],
            rotation=[45, 0, 45],
            energy=1000,
            size=2
        )
        
        # Fill light
        fill = create_light(
            light_type="AREA",
            name="Fill_Light",
            location=[-3, -3, 3],
            rotation=[60, 0, -30],
            energy=500,
            size=3
        )
        
        # Rim light
        rim = create_light(
            light_type="AREA",
            name="Rim_Light",
            location=[0, 5, 3],
            rotation=[-60, 0, 0],
            energy=800,
            size=1
        )
        
        results["lights"] = ["Key_Light", "Fill_Light", "Rim_Light"]
        
    elif lighting_preset == "sun_sky":
        sun = create_light(
            light_type="SUN",
            name="Sun",
            rotation=[45, 0, settings["world_settings"].get("sun_rotation", 0)],
            energy=5
        )
        results["lights"] = ["Sun"]
        
    elif lighting_preset == "studio":
        # Create studio lighting setup
        for i in range(3):
            angle = i * 120
            x = math.cos(math.radians(angle)) * 5
            y = math.sin(math.radians(angle)) * 5
            
            light = create_light(
                light_type="AREA",
                name=f"Studio_Light_{i+1}",
                location=[x, y, 4],
                energy=800,
                size=2
            )
        
        results["lights"] = [f"Studio_Light_{i+1}" for i in range(3)]
    
    # Setup world
    world = bpy.context.scene.world
    if world:
        world.use_nodes = True
        nodes = world.node_tree.nodes
        
        # Clear existing
        nodes.clear()
        
        # Add background node
        bg = nodes.new('ShaderNodeBackground')
        bg.location = (0, 0)
        
        # Add output
        output = nodes.new('ShaderNodeOutputWorld')
        output.location = (200, 0)
        
        # Set color
        bg_color = settings["world_settings"].get("background_color", [0.05, 0.05, 0.05])
        bg.inputs['Color'].default_value = (*bg_color, 1.0)
        bg.inputs['Strength'].default_value = 1.0
        
        # Connect
        world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])
    
    results["preset"] = preset
    results["created_at"] = datetime.now().isoformat()
    
    return results

@thread_safe
def create_hdri_environment(
    hdri_path: Optional[str] = None,
    strength: float = 1.0,
    rotation: float = 0.0
) -> Dict[str, Any]:
    """
    Setup HDRI environment lighting.
    
    Args:
        hdri_path: Path to HDRI image
        strength: Environment strength
        rotation: Z rotation in degrees
        
    Returns:
        HDRI setup results
    """
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Clear existing nodes
    nodes.clear()
    
    # Add nodes
    coord = nodes.new('ShaderNodeTexCoord')
    coord.location = (-800, 0)
    
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-600, 0)
    mapping.inputs['Rotation'].default_value[2] = math.radians(rotation)
    
    env_tex = nodes.new('ShaderNodeTexEnvironment')
    env_tex.location = (-400, 0)
    
    if hdri_path and os.path.exists(hdri_path):
        img = bpy.data.images.load(hdri_path)
        env_tex.image = img
    
    bg = nodes.new('ShaderNodeBackground')
    bg.location = (-200, 0)
    bg.inputs['Strength'].default_value = strength
    
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (0, 0)
    
    # Connect nodes
    links.new(coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
    links.new(env_tex.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], output.inputs['Surface'])
    
    return {
        "hdri_loaded": hdri_path is not None,
        "strength": strength,
        "rotation": rotation,
        "created_at": datetime.now().isoformat()
    }

# ============================================
# 📦 OBJECT CREATION & PRIMITIVES
# ============================================

@thread_safe
def create_mesh_object(
    primitive_type: str = "cube",
    size: float = 2.0,
    location: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    scale: Optional[List[float]] = None,
    name: Optional[str] = None,
    subdivisions: int = 0,
    segments: int = 32,
    rings: int = 16,
    vertices: int = 32,
    major_segments: int = 48,
    minor_segments: int = 12,
    ico_subdivisions: int = 2,
    x_subdivisions: int = 10,
    y_subdivisions: int = 10,
    collection: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create any mesh primitive object with full control.
    
    Args:
        primitive_type: Type of primitive (cube, sphere, cylinder, cone, torus, plane, ico_sphere, monkey, grid)
        size: Base size of the object
        location: 3D position [x, y, z]
        rotation: Rotation in degrees [x, y, z]
        scale: Scale factors [x, y, z]
        name: Custom name for the object
        subdivisions: Subdivision surface levels
        segments: Segments for UV sphere
        rings: Rings for UV sphere
        vertices: Vertices for cylinder/cone
        major_segments: Major segments for torus
        minor_segments: Minor segments for torus
        ico_subdivisions: Subdivisions for ico sphere
        x_subdivisions: X subdivisions for grid
        y_subdivisions: Y subdivisions for grid
        collection: Target collection name
    
    Returns:
        Complete object information including statistics
    """
    # Parameter validation and defaults
    location = location or [0, 0, 0]
    rotation = rotation or [0, 0, 0]
    scale = scale or [1, 1, 1]
    
    # Validate primitive type
    valid_primitives = ["cube", "sphere", "cylinder", "cone", "torus", 
                       "plane", "ico_sphere", "monkey", "grid"]
    if primitive_type not in valid_primitives:
        raise ValueError(f"Invalid primitive type. Must be one of: {valid_primitives}")
    
    # Store initial object count for tracking
    initial_objects = set(bpy.data.objects.keys())
    
    # Create primitive based on type
    creation_ops = {
        "cube": lambda: bpy.ops.mesh.primitive_cube_add(
            size=size, 
            location=location
        ),
        "sphere": lambda: bpy.ops.mesh.primitive_uv_sphere_add(
            radius=size/2,
            segments=segments,
            ring_count=rings,
            location=location
        ),
        "cylinder": lambda: bpy.ops.mesh.primitive_cylinder_add(
            radius=size/2,
            depth=size,
            vertices=vertices,
            location=location
        ),
        "cone": lambda: bpy.ops.mesh.primitive_cone_add(
            radius1=size/2,
            radius2=0,
            depth=size,
            vertices=vertices,
            location=location
        ),
        "torus": lambda: bpy.ops.mesh.primitive_torus_add(
            major_radius=size/2,
            minor_radius=size/4,
            major_segments=major_segments,
            minor_segments=minor_segments,
            location=location
        ),
        "plane": lambda: bpy.ops.mesh.primitive_plane_add(
            size=size,
            location=location
        ),
        "ico_sphere": lambda: bpy.ops.mesh.primitive_ico_sphere_add(
            radius=size/2,
            subdivisions=ico_subdivisions,
            location=location
        ),
        "monkey": lambda: bpy.ops.mesh.primitive_monkey_add(
            size=size,
            location=location
        ),
        "grid": lambda: bpy.ops.mesh.primitive_grid_add(
            x_subdivisions=x_subdivisions,
            y_subdivisions=y_subdivisions,
            size=size,
            location=location
        ),
    }
    
    # Execute creation
    creation_ops[primitive_type]()
    
    # Get the newly created object
    obj = bpy.context.active_object
    if not obj:
        new_objects = set(bpy.data.objects.keys()) - initial_objects
        if new_objects:
            obj = bpy.data.objects[new_objects.pop()]
        else:
            raise RuntimeError("Failed to create object")
    
    # Apply custom name
    if name:
        obj.name = name
        if obj.data:
            obj.data.name = f"{name}_mesh"
    
    # Apply transformations
    obj.rotation_euler = [math.radians(r) for r in rotation]
    obj.scale = scale
    
    # Apply subdivision surface if requested
    if subdivisions > 0:
        modifier = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        modifier.levels = subdivisions
        modifier.render_levels = subdivisions
        modifier.uv_smooth = 'PRESERVE_BOUNDARIES'
    
    # Move to collection if specified
    if collection:
        target_collection = bpy.data.collections.get(collection)
        if not target_collection:
            target_collection = bpy.data.collections.new(collection)
            bpy.context.scene.collection.children.link(target_collection)
        
        # Move object to target collection
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        target_collection.objects.link(obj)
    
    # Calculate statistics
    mesh_stats = {
        "vertices": len(obj.data.vertices),
        "edges": len(obj.data.edges),
        "faces": len(obj.data.polygons),
        "triangles": sum(len(p.vertices) - 2 for p in obj.data.polygons),
    }
    
    # Calculate bounding box
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    bbox_min = [min(v[i] for v in bbox) for i in range(3)]
    bbox_max = [max(v[i] for v in bbox) for i in range(3)]
    dimensions = [bbox_max[i] - bbox_min[i] for i in range(3)]
    
    return {
        "name": obj.name,
        "type": primitive_type,
        "location": list(obj.location),
        "rotation_degrees": rotation,
        "scale": list(obj.scale),
        "statistics": mesh_stats,
        "bounding_box": {
            "min": bbox_min,
            "max": bbox_max,
            "dimensions": dimensions,
            "volume": dimensions[0] * dimensions[1] * dimensions[2]
        },
        "subdivisions": subdivisions,
        "collection": collection or "Scene Collection",
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def create_curve_object(
    curve_type: str = "bezier",
    points: Optional[List[List[float]]] = None,
    name: Optional[str] = None,
    bevel_depth: float = 0.0,
    bevel_resolution: int = 4,
    resolution: int = 12,
    cyclic: bool = False,
    dimensions: str = "3D",
    fill_mode: str = "FULL",
    twist_mode: str = "MINIMUM",
    taper_object: Optional[str] = None,
    bevel_object: Optional[str] = None
) -> Dict[str, Any]:
    """Create curve objects (bezier, nurbs, path)."""
    
    # Default points if not provided
    if points is None:
        points = [[0, 0, 0], [2, 0, 1], [2, 2, 0], [0, 2, -1]]
    
    # Create curve data
    curve_data = bpy.data.curves.new(
        name=name or f"Curve_{curve_type}",
        type='CURVE'
    )
    
    # Configure curve properties
    curve_data.dimensions = dimensions
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = bevel_resolution
    curve_data.resolution_u = resolution
    curve_data.fill_mode = fill_mode
    curve_data.twist_mode = twist_mode
    
    # Create appropriate spline
    if curve_type == "bezier":
        spline = curve_data.splines.new('BEZIER')
        # Add bezier points (già ne ha uno di default)
        spline.bezier_points.add(len(points) - 1)
        
        for i, point in enumerate(points):
            bp = spline.bezier_points[i]
            bp.co = point
            # FIX: usa handle_left_type e handle_right_type
            bp.handle_left_type = 'AUTO'
            bp.handle_right_type = 'AUTO'
            bp.radius = 1.0
    
    elif curve_type == "nurbs":
        spline = curve_data.splines.new('NURBS')
        spline.points.add(len(points) - 1)
        for i, point in enumerate(points):
            spline.points[i].co = (*point, 1.0)
            spline.points[i].radius = 1.0
        spline.order_u = min(4, len(points))
        spline.use_endpoint_u = True
    
    else:  # poly/path
        spline = curve_data.splines.new('POLY')
        spline.points.add(len(points) - 1)
        for i, point in enumerate(points):
            spline.points[i].co = (*point, 1.0)
            spline.points[i].radius = 1.0
    
    spline.use_cyclic_u = cyclic
    
    # Create object
    obj = bpy.data.objects.new(name or f"Curve_{curve_type}", curve_data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    
    # Update to calculate length
    obj.data.update_tag()
    bpy.context.view_layer.update()
    
    # Get curve statistics
    total_length = 0
    # Calcola lunghezza solo se il metodo esiste
    if hasattr(spline, 'calc_length'):
        try:
            total_length = spline.calc_length()
        except:
            total_length = 0
    
    return {
        "name": obj.name,
        "type": f"curve_{curve_type}",
        "points": len(points),
        "cyclic": cyclic,
        "dimensions": dimensions,
        "bevel_depth": bevel_depth,
        "bevel_resolution": bevel_resolution,
        "resolution": resolution,
        "length": total_length,
        "fill_mode": fill_mode,
        "twist_mode": twist_mode,
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def create_text_object(
    text: str = "Hello Blender",
    font_size: float = 1.0,
    font_path: Optional[str] = None,
    extrude: float = 0.0,
    bevel_depth: float = 0.0,
    bevel_resolution: int = 4,
    location: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    name: Optional[str] = None,
    align_x: str = "LEFT",
    align_y: str = "TOP",
    character_spacing: float = 1.0,
    word_spacing: float = 1.0,
    line_spacing: float = 1.0,
    shear: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0
) -> Dict[str, Any]:
    """
    Create professional 3D text object with full typography control.
    
    Args:
        text: Text content (supports multiline)
        font_size: Size of the font
        font_path: Path to custom font file
        extrude: Extrusion depth
        bevel_depth: Bevel amount
        bevel_resolution: Bevel quality
        location: 3D position
        rotation: Rotation in degrees
        name: Object name
        align_x: Horizontal alignment (LEFT, CENTER, RIGHT, JUSTIFY, FLUSH)
        align_y: Vertical alignment (TOP, CENTER, BOTTOM)
        character_spacing: Character spacing multiplier
        word_spacing: Word spacing multiplier
        line_spacing: Line spacing multiplier
        shear: Italic shear amount
        offset_x: Horizontal offset
        offset_y: Vertical offset
        
    Returns:
        Complete text object information
    """
    location = location or [0, 0, 0]
    rotation = rotation or [0, 0, 0]
    
    # Create text curve
    font_curve = bpy.data.curves.new(type="FONT", name=name or "Text")
    
    # Set text content
    font_curve.body = text
    
    # Typography settings
    font_curve.size = font_size
    font_curve.space_character = character_spacing
    font_curve.space_word = word_spacing
    font_curve.space_line = line_spacing
    font_curve.shear = shear
    font_curve.offset_x = offset_x
    font_curve.offset_y = offset_y
    
    # 3D settings
    font_curve.extrude = extrude
    font_curve.bevel_depth = bevel_depth
    font_curve.bevel_resolution = bevel_resolution
    
    # Alignment
    font_curve.align_x = align_x
    font_curve.align_y = align_y
    
    # Load custom font if specified
    if font_path and os.path.exists(font_path):
        try:
            font = bpy.data.fonts.load(font_path)
            font_curve.font = font
        except Exception as e:
            logger.warning(f"Failed to load font {font_path}: {e}")
    
    # Create object
    obj = bpy.data.objects.new(name or "Text", font_curve)
    obj.location = location
    obj.rotation_euler = [math.radians(r) for r in rotation]
    
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    
    # Calculate text statistics
    line_count = text.count('\n') + 1
    char_count = len(text.replace('\n', ''))
    word_count = len(text.split())
    
    return {
        "name": obj.name,
        "text": text,
        "statistics": {
            "characters": char_count,
            "words": word_count,
            "lines": line_count
        },
        "font": {
            "size": font_size,
            "custom_font": font_path is not None,
            "shear": shear
        },
        "3d_properties": {
            "extrude": extrude,
            "bevel_depth": bevel_depth,
            "bevel_resolution": bevel_resolution
        },
        "alignment": {
            "horizontal": align_x,
            "vertical": align_y
        },
        "spacing": {
            "character": character_spacing,
            "word": word_spacing,
            "line": line_spacing
        },
        "location": list(obj.location),
        "rotation_degrees": rotation,
        "created_at": datetime.now().isoformat()
    }

# ============================================
# 🎯 OBJECT MANIPULATION
# ============================================

@thread_safe
def transform_object(
    object_name: str,
    location: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    scale: Optional[List[float]] = None,
    delta: bool = False,
    space: str = "WORLD",
    apply_transform: bool = False
) -> Dict[str, Any]:
    """
    Transform an object with advanced options.
    
    Args:
        object_name: Name of the object to transform
        location: New location [x, y, z] or delta if delta=True
        rotation: Rotation in degrees [x, y, z]
        scale: Scale factors [x, y, z]
        delta: If True, values are added to current transform
        space: Transform space (WORLD, LOCAL, PARENT)
        apply_transform: Apply the transformation permanently
        
    Returns:
        Complete transform information
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    # Store original transform for comparison
    original_transform = {
        "location": list(obj.location),
        "rotation": [math.degrees(r) for r in obj.rotation_euler],
        "scale": list(obj.scale)
    }
    
    # Apply location
    if location:
        if delta:
            obj.location += mathutils.Vector(location)
        else:
            if space == "WORLD":
                obj.location = location
            elif space == "LOCAL":
                obj.location = obj.matrix_world @ mathutils.Vector(location)
    
    # Apply rotation
    if rotation:
        rot_rad = [math.radians(r) for r in rotation]
        if delta:
            obj.rotation_euler[0] += rot_rad[0]
            obj.rotation_euler[1] += rot_rad[1]
            obj.rotation_euler[2] += rot_rad[2]
        else:
            obj.rotation_euler = rot_rad
    
    # Apply scale
    if scale:
        if delta:
            obj.scale[0] *= scale[0]
            obj.scale[1] *= scale[1]
            obj.scale[2] *= scale[2]
        else:
            obj.scale = scale
    
    # Apply transform if requested
    if apply_transform:
        bpy.context.view_layer.objects.active = obj
        if location:
            bpy.ops.object.location_clear(clear_delta=False)
        if rotation:
            bpy.ops.object.rotation_clear(clear_delta=False)
        if scale:
            bpy.ops.object.scale_clear(clear_delta=False)
    
    # Calculate transform matrix
    matrix = obj.matrix_world
    
    return {
        "name": obj.name,
        "original_transform": original_transform,
        "new_transform": {
            "location": list(obj.location),
            "rotation_degrees": [math.degrees(r) for r in obj.rotation_euler],
            "scale": list(obj.scale)
        },
        "delta_applied": delta,
        "space": space,
        "transform_applied": apply_transform,
        "matrix": [list(row) for row in matrix],
        "modified_at": datetime.now().isoformat()
    }

@thread_safe
def duplicate_object(
    object_name: str,
    linked: bool = False,
    offset: Optional[List[float]] = None,
    count: int = 1,
    collection: Optional[str] = None,
    recursive: bool = True
) -> Dict[str, Any]:
    """
    Duplicate objects with advanced options.
    
    Args:
        object_name: Name of object to duplicate
        linked: Create linked duplicate (instance)
        offset: Position offset for each duplicate
        count: Number of duplicates to create
        collection: Target collection for duplicates
        recursive: Also duplicate child objects
        
    Returns:
        Detailed duplication information
    """
    offset = offset or [0, 0, 0]
    
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    duplicates = []
    
    for i in range(count):
        # Calculate offset for this duplicate
        current_offset = mathutils.Vector([
            offset[0] * (i + 1),
            offset[1] * (i + 1),
            offset[2] * (i + 1)
        ])
        
        # Duplicate object
        new_obj = obj.copy()
        
        # Duplicate data if not linked
        if not linked and obj.data:
            new_obj.data = obj.data.copy()
            new_obj.data.name = f"{obj.data.name}_copy_{i+1}"
        
        new_obj.location = obj.location + current_offset
        new_obj.name = f"{obj.name}_copy_{i+1}"
        
        # Add to collection
        if collection:
            target_coll = bpy.data.collections.get(collection)
            if not target_coll:
                target_coll = bpy.data.collections.new(collection)
                bpy.context.scene.collection.children.link(target_coll)
            target_coll.objects.link(new_obj)
        else:
            bpy.context.collection.objects.link(new_obj)
        
        # Duplicate children if recursive
        if recursive and obj.children:
            for child in obj.children:
                child_dup = duplicate_object(
                    child.name,
                    linked=linked,
                    offset=[0, 0, 0],
                    count=1,
                    collection=collection,
                    recursive=True
                )
                if child_dup['duplicates']:
                    child_obj = bpy.data.objects.get(child_dup['duplicates'][0]['name'])
                    if child_obj:
                        child_obj.parent = new_obj
                        child_obj.location = child.location
        
        duplicates.append({
            "name": new_obj.name,
            "linked": linked,
            "location": list(new_obj.location)
        })
    
    return {
        "original": object_name,
        "duplicates": duplicates,
        "count": count,
        "linked": linked,
        "recursive": recursive,
        "collection": collection,
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def delete_objects(
    object_names: List[str],
    delete_data: bool = True,
    delete_hierarchy: bool = False,
    confirm: bool = True
) -> Dict[str, Any]:
    """
    Delete objects with comprehensive options.
    
    Args:
        object_names: List of object names to delete
        delete_data: Also delete mesh/curve data
        delete_hierarchy: Delete children as well
        confirm: Safety confirmation (set False to force delete)
        
    Returns:
        Detailed deletion report
    """
    if not confirm:
        logger.warning("Deleting objects without confirmation")
    
    deleted = []
    not_found = []
    data_deleted = []
    children_deleted = []
    
    for name in object_names:
        obj = bpy.data.objects.get(name)
        if not obj:
            not_found.append(name)
            continue
        
        # Collect children if deleting hierarchy
        if delete_hierarchy:
            def get_all_children(obj):
                children = []
                for child in obj.children:
                    children.append(child.name)
                    children.extend(get_all_children(child))
                return children
            
            child_names = get_all_children(obj)
            children_deleted.extend(child_names)
            
            # Delete children first
            for child_name in child_names:
                child = bpy.data.objects.get(child_name)
                if child:
                    bpy.data.objects.remove(child)
        
        # Store data reference before deletion
        obj_data = obj.data
        obj_data_name = obj_data.name if obj_data else None
        
        # Delete the object
        bpy.data.objects.remove(obj)
        deleted.append(name)
        
        # Delete orphaned data if requested
        if delete_data and obj_data:
            if obj_data.users == 0:
                if hasattr(bpy.data, 'meshes') and obj_data in bpy.data.meshes.values():
                    bpy.data.meshes.remove(obj_data)
                    data_deleted.append(('mesh', obj_data_name))
                elif hasattr(bpy.data, 'curves') and obj_data in bpy.data.curves.values():
                    bpy.data.curves.remove(obj_data)
                    data_deleted.append(('curve', obj_data_name))
                elif hasattr(bpy.data, 'cameras') and obj_data in bpy.data.cameras.values():
                    bpy.data.cameras.remove(obj_data)
                    data_deleted.append(('camera', obj_data_name))
                elif hasattr(bpy.data, 'lights') and obj_data in bpy.data.lights.values():
                    bpy.data.lights.remove(obj_data)
                    data_deleted.append(('light', obj_data_name))
    
    return {
        "deleted": deleted,
        "not_found": not_found,
        "data_deleted": data_deleted,
        "children_deleted": children_deleted,
        "total_deleted": len(deleted) + len(children_deleted),
        "total_data_deleted": len(data_deleted),
        "deleted_at": datetime.now().isoformat()
    }

# ============================================
# 🔧 MODIFIERS SYSTEM
# ============================================

@thread_safe
def add_modifier(
    object_name: str,
    modifier_type: str,
    settings: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    show_viewport: bool = True,
    show_render: bool = True
) -> Dict[str, Any]:
    """
    Add any modifier to an object with full configuration.
    
    Args:
        object_name: Target object
        modifier_type: Type of modifier (SUBSURF, ARRAY, MIRROR, etc.)
        settings: Modifier-specific settings
        name: Custom name for modifier
        show_viewport: Show in viewport
        show_render: Show in render
        
    Returns:
        Complete modifier information
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    # Validate modifier type
    valid_modifiers = [
        'ARRAY', 'BEVEL', 'BOOLEAN', 'BUILD', 'DECIMATE', 'EDGE_SPLIT',
        'MASK', 'MIRROR', 'MULTIRES', 'REMESH', 'SCREW', 'SKIN',
        'SOLIDIFY', 'SUBSURF', 'TRIANGULATE', 'WELD', 'WIREFRAME',
        'ARMATURE', 'CAST', 'CURVE', 'DISPLACE', 'HOOK', 'LAPLACIANDEFORM',
        'LAPLACIANSMOOTH', 'LATTICE', 'MESH_DEFORM', 'SHRINKWRAP',
        'SIMPLE_DEFORM', 'SMOOTH', 'CORRECTIVE_SMOOTH', 'SURFACE_DEFORM',
        'WARP', 'WAVE', 'CLOTH', 'COLLISION', 'DYNAMIC_PAINT', 'EXPLODE',
        'FLUID', 'OCEAN', 'PARTICLE_INSTANCE', 'PARTICLE_SYSTEM',
        'SOFT_BODY', 'SURFACE', 'MESH_CACHE', 'MESH_SEQUENCE_CACHE',
        'NODES', 'NORMAL_EDIT', 'UV_PROJECT', 'UV_WARP', 'VERTEX_WEIGHT_EDIT',
        'VERTEX_WEIGHT_MIX', 'VERTEX_WEIGHT_PROXIMITY'
    ]
    
    if modifier_type not in valid_modifiers:
        raise ValueError(f"Invalid modifier type. Must be one of: {valid_modifiers}")
    
    # Add modifier
    modifier = obj.modifiers.new(
        name=name or modifier_type,
        type=modifier_type
    )
    
    # Configure visibility
    modifier.show_viewport = show_viewport
    modifier.show_render = show_render
    
    # Apply settings based on modifier type
    settings = settings or {}
    
    # Type-specific configurations
    if modifier_type == "SUBSURF":
        modifier.levels = settings.get('levels', 2)
        modifier.render_levels = settings.get('render_levels', 2)
        modifier.subdivision_type = settings.get('subdivision_type', 'CATMULL_CLARK')
        modifier.use_creases = settings.get('use_creases', True)
        modifier.quality = settings.get('quality', 3)
        
    elif modifier_type == "ARRAY":
        modifier.count = settings.get('count', 3)
        modifier.use_constant_offset = settings.get('use_constant_offset', False)
        modifier.constant_offset_displace = settings.get('constant_offset', [0, 0, 0])
        modifier.use_relative_offset = settings.get('use_relative_offset', True)
        modifier.relative_offset_displace[0] = settings.get('offset_x', 1.1)
        modifier.relative_offset_displace[1] = settings.get('offset_y', 0)
        modifier.relative_offset_displace[2] = settings.get('offset_z', 0)
        modifier.use_object_offset = settings.get('use_object_offset', False)
        if settings.get('offset_object'):
            offset_obj = bpy.data.objects.get(settings['offset_object'])
            if offset_obj:
                modifier.offset_object = offset_obj
        
    elif modifier_type == "MIRROR":
        modifier.use_axis[0] = settings.get('x', True)
        modifier.use_axis[1] = settings.get('y', False)
        modifier.use_axis[2] = settings.get('z', False)
        modifier.use_bisect_axis[0] = settings.get('bisect_x', False)
        modifier.use_bisect_axis[1] = settings.get('bisect_y', False)
        modifier.use_bisect_axis[2] = settings.get('bisect_z', False)
        modifier.use_clip = settings.get('use_clip', True)
        modifier.merge_threshold = settings.get('merge_threshold', 0.001)
        if settings.get('mirror_object'):
            mirror_obj = bpy.data.objects.get(settings['mirror_object'])
            if mirror_obj:
                modifier.mirror_object = mirror_obj
        
    elif modifier_type == "SOLIDIFY":
        modifier.thickness = settings.get('thickness', 0.1)
        modifier.offset = settings.get('offset', -1.0)
        modifier.use_even_offset = settings.get('use_even_offset', True)
        modifier.use_quality_normals = settings.get('use_quality_normals', True)
        modifier.use_rim = settings.get('use_rim', True)
        modifier.use_rim_only = settings.get('use_rim_only', False)
        
    elif modifier_type == "BEVEL":
        modifier.width = settings.get('width', 0.1)
        modifier.segments = settings.get('segments', 3)
        modifier.limit_method = settings.get('limit_method', 'ANGLE')
        if modifier.limit_method == 'ANGLE':
            modifier.angle_limit = math.radians(settings.get('angle', 30))
        modifier.offset_type = settings.get('offset_type', 'OFFSET')
        modifier.profile = settings.get('profile', 0.5)
        modifier.use_clamp_overlap = settings.get('use_clamp_overlap', True)
        
    elif modifier_type == "BOOLEAN":
        modifier.operation = settings.get('operation', 'DIFFERENCE')
        modifier.solver = settings.get('solver', 'FAST')
        if settings.get('object'):
            bool_obj = bpy.data.objects.get(settings['object'])
            if bool_obj:
                modifier.object = bool_obj
        modifier.use_self = settings.get('use_self', False)
        modifier.use_hole_tolerant = settings.get('use_hole_tolerant', False)
        
    # Apply any remaining generic settings
    for key, value in settings.items():
        if hasattr(modifier, key):
            try:
                setattr(modifier, key, value)
            except Exception as e:
                logger.warning(f"Could not set {key} on modifier: {e}")
    
    return {
        "object": object_name,
        "modifier": modifier.name,
        "type": modifier_type,
        "settings_applied": settings,
        "show_viewport": show_viewport,
        "show_render": show_render,
        "created_at": datetime.now().isoformat()
    }

@thread_safe
def apply_modifier(
    object_name: str,
    modifier_name: str,
    apply_as: str = "DATA",
    keep_original: bool = False
) -> Dict[str, Any]:
    """
    Apply a modifier with advanced options.
    
    Args:
        object_name: Object name
        modifier_name: Modifier to apply
        apply_as: How to apply (DATA, SHAPE)
        keep_original: Keep a copy of the original object
        
    Returns:
        Detailed application result
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    modifier = obj.modifiers.get(modifier_name)
    if not modifier:
        raise ValueError(f"Modifier '{modifier_name}' not found on object '{object_name}'")
    
    # Statistics before applying
    stats_before = {
        "vertices": len(obj.data.vertices) if hasattr(obj.data, 'vertices') else 0,
        "edges": len(obj.data.edges) if hasattr(obj.data, 'edges') else 0,
        "faces": len(obj.data.polygons) if hasattr(obj.data, 'polygons') else 0
    }
    
    # Keep original if requested
    original_name = None
    if keep_original:
        original = duplicate_object(object_name, linked=False, offset=[0, 0, 0])
        original_name = original['duplicates'][0]['name']
        original_obj = bpy.data.objects.get(original_name)
        if original_obj:
            original_obj.name = f"{object_name}_original"
            original_obj.hide_set(True)
            original_name = original_obj.name
    
    # Select and make active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Apply modifier
    try:
        if apply_as == "SHAPE":
            bpy.ops.object.modifier_apply_as_shapekey(modifier=modifier_name)
        else:
            bpy.ops.object.modifier_apply(modifier=modifier_name)
    except Exception as e:
        if keep_original and original_name:
            # Remove the backup if apply failed
            delete_objects([original_name], delete_data=True)
        raise RuntimeError(f"Failed to apply modifier: {e}")
    
    # Statistics after applying
    stats_after = {
        "vertices": len(obj.data.vertices) if hasattr(obj.data, 'vertices') else 0,
        "edges": len(obj.data.edges) if hasattr(obj.data, 'edges') else 0,
        "faces": len(obj.data.polygons) if hasattr(obj.data, 'polygons') else 0
    }
    
    return {
        "object": object_name,
        "applied_modifier": modifier_name,
        "apply_method": apply_as,
        "statistics": {
            "before": stats_before,
            "after": stats_after,
            "difference": {
                "vertices": stats_after["vertices"] - stats_before["vertices"],
                "edges": stats_after["edges"] - stats_before["edges"],
                "faces": stats_after["faces"] - stats_before["faces"]
            }
        },
        "original_backup": original_name,
        "applied_at": datetime.now().isoformat()
    }

# ============================================
# 🎨 MATERIALS & SHADING
# ============================================

@thread_safe
def create_material(
    name: str,
    material_type: str = "principled",
    base_color: Optional[List[float]] = None,
    metallic: float = 0.0,
    roughness: float = 0.5,
    ior: float = 1.45,
    transmission: float = 0.0,
    emission_color: Optional[List[float]] = None,
    emission_strength: float = 0.0,
    alpha: float = 1.0,
    use_backface_culling: bool = False,
    blend_mode: str = "OPAQUE",
    use_nodes: bool = True
) -> Dict[str, Any]:
    """
    Create production-ready materials with full PBR support.
    
    Args:
        name: Material name
        material_type: Shader type (principled, emission, glass, glossy, transparent)
        base_color: Base color RGBA
        metallic: Metallic value (0-1)
        roughness: Roughness value (0-1)
        ior: Index of refraction
        transmission: Transmission for glass (0-1)
        emission_color: Emission color RGBA
        emission_strength: Emission strength
        alpha: Alpha transparency
        use_backface_culling: Enable backface culling
        blend_mode: Blend mode (OPAQUE, CLIP, BLEND)
        use_nodes: Use node-based shading
        
    Returns:
        Complete material information
    """
    base_color = base_color or [0.8, 0.8, 0.8, 1.0]
    emission_color = emission_color or [0, 0, 0, 1]
    
    # Create material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = use_nodes
    mat.use_backface_culling = use_backface_culling
    mat.blend_method = blend_mode
    
    if use_nodes:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear existing nodes
        nodes.clear()
        
        # Add output node
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (400, 0)
        
        # Create shader based on type
        if material_type == "principled":
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf.location = (0, 0)
            
            # Set all principled BSDF parameters with compatibility checks
            # Base properties (always available)
            bsdf.inputs['Base Color'].default_value = base_color
            bsdf.inputs['Metallic'].default_value = metallic
            bsdf.inputs['Roughness'].default_value = roughness
            
            # IOR - check different naming conventions
            if 'IOR' in bsdf.inputs:
                bsdf.inputs['IOR'].default_value = ior
            elif 'Index of Refraction' in bsdf.inputs:
                bsdf.inputs['Index of Refraction'].default_value = ior
            
            # Transmission - Blender 4.0+ uses "Transmission Weight"
            if transmission > 0:
                if 'Transmission' in bsdf.inputs:
                    bsdf.inputs['Transmission'].default_value = transmission
                elif 'Transmission Weight' in bsdf.inputs:
                    bsdf.inputs['Transmission Weight'].default_value = transmission
            
            # Alpha
            if 'Alpha' in bsdf.inputs:
                bsdf.inputs['Alpha'].default_value = alpha
            
            # Emission - check for both old and new naming
            if emission_strength > 0:
                # Blender 4.0+ separates emission into color and strength
                if 'Emission Color' in bsdf.inputs:
                    bsdf.inputs['Emission Color'].default_value = emission_color
                    if 'Emission Strength' in bsdf.inputs:
                        bsdf.inputs['Emission Strength'].default_value = emission_strength
                elif 'Emission' in bsdf.inputs:
                    # Older Blender versions
                    bsdf.inputs['Emission'].default_value = emission_color[:3] + [1.0]
                    if 'Emission Strength' in bsdf.inputs:
                        bsdf.inputs['Emission Strength'].default_value = emission_strength
            
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            
        elif material_type == "emission":
            emission = nodes.new(type='ShaderNodeEmission')
            emission.location = (0, 0)
            emission.inputs['Color'].default_value = emission_color
            emission.inputs['Strength'].default_value = max(emission_strength, 1.0)
            links.new(emission.outputs['Emission'], output.inputs['Surface'])
            
        elif material_type == "glass":
            glass = nodes.new(type='ShaderNodeBsdfGlass')
            glass.location = (0, 0)
            glass.inputs['Color'].default_value = base_color
            glass.inputs['Roughness'].default_value = roughness
            
            # IOR compatibility
            if 'IOR' in glass.inputs:
                glass.inputs['IOR'].default_value = ior
            elif 'Index of Refraction' in glass.inputs:
                glass.inputs['Index of Refraction'].default_value = ior
                
            links.new(glass.outputs['BSDF'], output.inputs['Surface'])
            
        elif material_type == "glossy":
            glossy = nodes.new(type='ShaderNodeBsdfGlossy')
            glossy.location = (0, 0)
            glossy.inputs['Color'].default_value = base_color
            glossy.inputs['Roughness'].default_value = roughness
            links.new(glossy.outputs['BSDF'], output.inputs['Surface'])
            
        elif material_type == "transparent":
            transparent = nodes.new(type='ShaderNodeBsdfTransparent')
            transparent.location = (0, 0)
            transparent.inputs['Color'].default_value = base_color
            links.new(transparent.outputs['BSDF'], output.inputs['Surface'])
    
    # Generate unique ID for material
    material_id = hashlib.md5(name.encode()).hexdigest()[:8]
    
    return {
        "name": mat.name,
        "id": material_id,
        "type": material_type,
        "properties": {
            "base_color": base_color,
            "metallic": metallic,
            "roughness": roughness,
            "ior": ior,
            "transmission": transmission,
            "alpha": alpha,
            "emission_color": emission_color,
            "emission_strength": emission_strength
        },
        "settings": {
            "use_nodes": use_nodes,
            "use_backface_culling": use_backface_culling,
            "blend_mode": blend_mode
        },
        "node_tree": mat.node_tree.name if use_nodes else None,
        "created_at": datetime.now().isoformat()
    }
@thread_safe  
def assign_material(
    object_name: str,
    material_name: str,
    slot_index: int = -1,
    assign_to: str = "OBJECT"
) -> Dict[str, Any]:
    """
    Assign material with advanced options.
    
    Args:
        object_name: Target object
        material_name: Material to assign
        slot_index: Material slot index (-1 for new slot)
        assign_to: Assignment target (OBJECT, DATA)
        
    Returns:
        Complete assignment information
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    mat = bpy.data.materials.get(material_name)
    if not mat:
        raise ValueError(f"Material '{material_name}' not found")
    
    # Check if object can have materials
    if not hasattr(obj.data, 'materials'):
        raise ValueError(f"Object '{object_name}' cannot have materials")
    
    # Store previous material for reference
    previous_material = None
    
    if slot_index == -1:
        # Add new slot
        obj.data.materials.append(mat)
        slot_index = len(obj.material_slots) - 1
    else:
        # Replace existing slot
        if slot_index < len(obj.material_slots):
            previous_material = obj.material_slots[slot_index].material
            if assign_to == "OBJECT":
                obj.material_slots[slot_index].material = mat
            else:
                obj.data.materials[slot_index] = mat
        else:
            # Extend slots if necessary
            while len(obj.data.materials) <= slot_index:
                obj.data.materials.append(None)
            obj.data.materials[slot_index] = mat
    
    return {
        "object": object_name,
        "material": material_name,
        "slot_index": slot_index,
        "total_slots": len(obj.material_slots),
        "previous_material": previous_material.name if previous_material else None,
        "assign_to": assign_to,
        "assigned_at": datetime.now().isoformat()
    }

# ============================================
# 💡 LIGHTING SYSTEM
# ============================================

@thread_safe
def create_light(
    light_type: str = "POINT",
    name: Optional[str] = None,
    location: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    energy: float = 1000.0,
    color: Optional[List[float]] = None,
    size: float = 0.25,
    spot_size: float = 45.0,
    spot_blend: float = 0.15,
    shadow_soft_size: float = 0.25,
    use_shadow: bool = True,
    use_contact_shadow: bool = False
) -> Dict[str, Any]:
    """
    Create professional lighting with full control.
    
    Args:
        light_type: Type of light (POINT, SUN, SPOT, AREA)
        name: Light name
        location: 3D position
        rotation: Rotation in degrees
        energy: Light power/energy
        color: Light color RGB
        size: Light size (for area/point lights)
        spot_size: Spot cone angle in degrees
        spot_blend: Spot edge softness
        shadow_soft_size: Shadow softness
        use_shadow: Enable shadows
        use_contact_shadow: Enable contact shadows
        
    Returns:
        Complete light information
    """
    location = location or [0, 0, 5]
    rotation = rotation or [0, 0, 0]
    color = color or [1, 1, 1]
    
    # Validate light type
    valid_types = ["POINT", "SUN", "SPOT", "AREA"]
    if light_type not in valid_types:
        raise ValueError(f"Invalid light type. Must be one of: {valid_types}")
    
    # Create light data
    light_data = bpy.data.lights.new(
        name=name or f"Light_{light_type}",
        type=light_type
    )
    
    # Configure light properties
    light_data.energy = energy
    light_data.color = color
    light_data.use_shadow = use_shadow
    light_data.use_contact_shadow = use_contact_shadow
    
    # Type-specific settings
    if light_type == "POINT":
        light_data.shadow_soft_size = shadow_soft_size
        
    elif light_type == "SUN":
        light_data.angle = math.radians(size)
        
    elif light_type == "SPOT":
        light_data.spot_size = math.radians(spot_size)
        light_data.spot_blend = spot_blend
        light_data.shadow_soft_size = shadow_soft_size
        
    elif light_type == "AREA":
        light_data.shape = 'SQUARE'
        light_data.size = size
    
    # Create light object
    light_obj = bpy.data.objects.new(
        name=name or f"Light_{light_type}",
        object_data=light_data
    )
    light_obj.location = location
    light_obj.rotation_euler = [math.radians(r) for r in rotation]
    
    bpy.context.collection.objects.link(light_obj)
    
    # Calculate illumination power
    luminous_power = energy * (color[0] + color[1] + color[2]) / 3
    
    return {
        "name": light_obj.name,
        "type": light_type,
        "location": list(light_obj.location),
        "rotation_degrees": rotation,
        "properties": {
            "energy": energy,
            "color": color,
            "luminous_power": luminous_power,
            "size": size,
            "spot_size": spot_size if light_type == "SPOT" else None,
            "spot_blend": spot_blend if light_type == "SPOT" else None
        },
        "shadows": {
            "enabled": use_shadow,
            "contact_shadows": use_contact_shadow,
            "softness": shadow_soft_size
        },
        "created_at": datetime.now().isoformat()
    }

# ============================================
# 📷 CAMERA SYSTEM
# ============================================

@thread_safe
def create_camera(
    name: Optional[str] = None,
    location: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    camera_type: str = "PERSP",
    focal_length: float = 50.0,
    sensor_width: float = 36.0,
    sensor_height: float = 24.0,
    sensor_fit: str = "AUTO",
    depth_of_field: bool = False,
    focus_object: Optional[str] = None,
    focus_distance: float = 10.0,
    f_stop: float = 2.8,
    blade_count: int = 0,
    blade_rotation: float = 0.0,
    orthographic_scale: float = 6.0,
    clip_start: float = 0.1,
    clip_end: float = 1000.0,
    passepartout_alpha: float = 0.5
) -> Dict[str, Any]:
    """
    Create professional camera with complete settings.
    
    Args:
        name: Camera name
        location: 3D position
        rotation: Rotation in degrees
        camera_type: Camera type (PERSP, ORTHO, PANO)
        focal_length: Lens focal length in mm
        sensor_width: Sensor width in mm
        sensor_height: Sensor height in mm
        sensor_fit: Sensor fit mode (AUTO, HORIZONTAL, VERTICAL)
        depth_of_field: Enable DoF
        focus_object: Object name to focus on
        focus_distance: Manual focus distance
        f_stop: Aperture f-stop
        blade_count: Aperture blade count (0 for circle)
        blade_rotation: Aperture blade rotation
        orthographic_scale: Scale for orthographic camera
        clip_start: Near clipping distance
        clip_end: Far clipping distance
        passepartout_alpha: Viewport passepartout opacity
        
    Returns:
        Complete camera information
    """
    location = location or [7, -7, 5]
    rotation = rotation or [60, 0, 45]
    
    # Create camera data
    cam_data = bpy.data.cameras.new(name=name or "Camera")
    
    # Basic settings
    cam_data.type = camera_type
    cam_data.lens = focal_length
    cam_data.sensor_width = sensor_width
    cam_data.sensor_height = sensor_height
    cam_data.sensor_fit = sensor_fit
    cam_data.clip_start = clip_start
    cam_data.clip_end = clip_end
    cam_data.passepartout_alpha = passepartout_alpha
    
    # Orthographic settings
    if camera_type == "ORTHO":
        cam_data.ortho_scale = orthographic_scale
    
    # Depth of field settings
    if depth_of_field:
        cam_data.dof.use_dof = True
        cam_data.dof.aperture_fstop = f_stop
        cam_data.dof.aperture_blades = blade_count
        cam_data.dof.aperture_rotation = math.radians(blade_rotation)
        
        if focus_object:
            focus_obj = bpy.data.objects.get(focus_object)
            if focus_obj:
                cam_data.dof.focus_object = focus_obj
        else:
            cam_data.dof.focus_distance = focus_distance
    
    # Create camera object
    cam_obj = bpy.data.objects.new(name or "Camera", cam_data)
    cam_obj.location = location
    cam_obj.rotation_euler = [math.radians(r) for r in rotation]
    
    bpy.context.collection.objects.link(cam_obj)
    
    # Calculate field of view
    fov_horizontal = 2 * math.atan(sensor_width / (2 * focal_length))
    fov_vertical = 2 * math.atan(sensor_height / (2 * focal_length))
    
    return {
        "name": cam_obj.name,
        "type": camera_type,
        "location": list(cam_obj.location),
        "rotation_degrees": rotation,
        "lens": {
            "focal_length": focal_length,
            "fov_horizontal": math.degrees(fov_horizontal),
            "fov_vertical": math.degrees(fov_vertical)
        },
        "sensor": {
            "width": sensor_width,
            "height": sensor_height,
            "fit": sensor_fit,
            "aspect_ratio": sensor_width / sensor_height
        },
        "depth_of_field": {
            "enabled": depth_of_field,
            "f_stop": f_stop if depth_of_field else None,
            "focus_object": focus_object,
            "focus_distance": focus_distance if not focus_object else None,
            "blades": blade_count if depth_of_field else None
        },
        "clipping": {
            "start": clip_start,
            "end": clip_end
        },
        "created_at": datetime.now().isoformat()
    }

# ============================================
# 🎬 ANIMATION SYSTEM
# ============================================

@thread_safe
def create_keyframe(
    object_name: str,
    frame: int,
    property_path: str = "location",
    values: Optional[List[float]] = None,
    interpolation: str = "BEZIER",
    easing: str = "AUTO",
    handle_type: str = "AUTO"
) -> Dict[str, Any]:
    """
    Create advanced animation keyframe.
    
    Args:
        object_name: Object to animate
        frame: Frame number
        property_path: Property to animate
        values: Property values
        interpolation: Interpolation type (BEZIER, LINEAR, CONSTANT, BOUNCE, etc.)
        easing: Easing type (AUTO, EASE_IN, EASE_OUT, EASE_IN_OUT)
        handle_type: Handle type for bezier (AUTO, VECTOR, ALIGNED, FREE)
        
    Returns:
        Complete keyframe information
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    # Set frame
    bpy.context.scene.frame_set(frame)
    
    # Apply values if provided
    if values is not None:
        if property_path == "location":
            obj.location = values
        elif property_path == "rotation_euler":
            obj.rotation_euler = [math.radians(v) for v in values]
        elif property_path == "scale":
            obj.scale = values
        else:
            # Handle custom properties
            if "." in property_path:
                # Nested property
                parts = property_path.split(".")
                target = obj
                for part in parts[:-1]:
                    target = getattr(target, part)
                setattr(target, parts[-1], values)
            else:
                setattr(obj, property_path, values)
    
    # Insert keyframe
    obj.keyframe_insert(data_path=property_path, frame=frame)
    
    # Configure keyframe properties
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            if fcurve.data_path == property_path:
                for keyframe in fcurve.keyframe_points:
                    if keyframe.co[0] == frame:
                        keyframe.interpolation = interpolation
                        keyframe.easing = easing
                        keyframe.handle_left_type = handle_type
                        keyframe.handle_right_type = handle_type
    
    # Get actual values
    actual_values = None
    if property_path == "location":
        actual_values = list(obj.location)
    elif property_path == "rotation_euler":
        actual_values = [math.degrees(r) for r in obj.rotation_euler]
    elif property_path == "scale":
        actual_values = list(obj.scale)
    else:
        try:
            actual_values = getattr(obj, property_path)
            if hasattr(actual_values, '__iter__'):
                actual_values = list(actual_values)
        except:
            actual_values = values
    
    return {
        "object": object_name,
        "frame": frame,
        "property": property_path,
        "values": actual_values,
        "interpolation": interpolation,
        "easing": easing,
        "handle_type": handle_type,
        "created_at": datetime.now().isoformat()
    }

# ============================================
# 🧭 SPATIAL AWARENESS FOR LLM
# ============================================

def get_object_position(object_name: str) -> Dict[str, Any]:
    """
    Get exact position and bounds of an object.
    
    Returns:
        Dict with location, center, size, bounding_box
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        raise ValueError(f"Object '{object_name}' not found")
    
    # Calcola bounding box in world space
    bbox_world = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    center = sum(bbox_world, mathutils.Vector()) / 8
    
    min_co = [min(v[i] for v in bbox_world) for i in range(3)]
    max_co = [max(v[i] for v in bbox_world) for i in range(3)]
    size = [max_co[i] - min_co[i] for i in range(3)]
    
    return {
        "object": object_name,
        "location": list(obj.location),  # Origin point
        "center": list(center),          # True center
        "size": size,                    # [width_x, depth_y, height_z]
        "radius": max(size) / 2,
        "bounding_box": {
            "min": min_co,
            "max": max_co
        },
        "front": [center[0], max_co[1], center[2]],  # Front center point
        "back": [center[0], min_co[1], center[2]],   # Back center point
        "top": [center[0], center[1], max_co[2]],    # Top center point
        "bottom": [center[0], center[1], min_co[2]], # Bottom center point
        "left": [min_co[0], center[1], center[2]],   # Left center point
        "right": [max_co[0], center[1], center[2]]   # Right center point
    }

def calculate_position_relative_to(
    reference_object: str,
    direction: str = "front",  # front, back, left, right, top, bottom, center
    distance: float = 1.0,
    offset: Optional[List[float]] = None  # Additional [x, y, z] offset
) -> List[float]:
    """
    Calculate a position relative to another object.
    
    Args:
        reference_object: Name of reference object
        direction: Direction relative to object (front, back, left, right, top, bottom, center)
        distance: Distance from object surface
        offset: Additional offset [x, y, z]
        
    Returns:
        Position coordinates [x, y, z]
    """
    obj_info = get_object_position(reference_object)
    center = obj_info["center"]
    size = obj_info["size"]
    
    offset = offset or [0, 0, 0]
    
    # Calculate base position based on direction
    positions = {
        "front": [center[0], center[1] + size[1]/2 + distance, center[2]],
        "back": [center[0], center[1] - size[1]/2 - distance, center[2]],
        "left": [center[0] - size[0]/2 - distance, center[1], center[2]],
        "right": [center[0] + size[0]/2 + distance, center[1], center[2]],
        "top": [center[0], center[1], center[2] + size[2]/2 + distance],
        "bottom": [center[0], center[1], center[2] - size[2]/2 - distance],
        "center": center.copy()
    }
    
    if direction not in positions:
        raise ValueError(f"Invalid direction. Use: {list(positions.keys())}")
    
    position = positions[direction]
    
    # Apply additional offset
    return [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]

def get_all_objects_positions() -> Dict[str, Any]:
    """
    Get positions of all objects in the scene.
    
    Returns:
        Dictionary with all objects and their positions
    """
    objects_map = {}
    
    for obj in bpy.data.objects:
        objects_map[obj.name] = {
            "type": obj.type,
            "location": list(obj.location),
            "size": [
                abs(max(v[0] for v in obj.bound_box) - min(v[0] for v in obj.bound_box)),
                abs(max(v[1] for v in obj.bound_box) - min(v[1] for v in obj.bound_box)),
                abs(max(v[2] for v in obj.bound_box) - min(v[2] for v in obj.bound_box))
            ] if obj.bound_box else [0, 0, 0]
        }
    
    # Calcola anche relazioni spaziali semplici
    if len(objects_map) > 1:
        relationships = []
        obj_names = list(objects_map.keys())
        
        for i, obj_a in enumerate(obj_names[:5]):  # Limita a 5 oggetti per semplicità
            for obj_b in obj_names[i+1:i+3]:  # Max 2 relazioni per oggetto
                loc_a = mathutils.Vector(objects_map[obj_a]["location"])
                loc_b = mathutils.Vector(objects_map[obj_b]["location"])
                
                diff = loc_b - loc_a
                distance = diff.length
                
                # Determina relazione principale
                if abs(diff.x) > abs(diff.y) and abs(diff.x) > abs(diff.z):
                    relation = f"{obj_b} is {'right of' if diff.x > 0 else 'left of'} {obj_a}"
                elif abs(diff.y) > abs(diff.z):
                    relation = f"{obj_b} is {'in front of' if diff.y > 0 else 'behind'} {obj_a}"
                else:
                    relation = f"{obj_b} is {'above' if diff.z > 0 else 'below'} {obj_a}"
                
                relationships.append(f"{relation} (distance: {distance:.2f})")
    else:
        relationships = []
    
    return {
        "total_objects": len(objects_map),
        "objects": objects_map,
        "relationships": relationships[:10],  # Max 10 relazioni
        "coordinate_system": {
            "x": "right/left (-x = left, +x = right)",
            "y": "forward/back (-y = back, +y = forward/front)",
            "z": "up/down (-z = down, +z = up)"
        }
    }

def find_empty_space(
    size_needed: float = 1.0,
    preferred_height: float = 0.0,
    avoid_center: bool = False
) -> List[float]:
    """
    Find empty space in the scene for a new object.
    
    Args:
        size_needed: Radius of space needed
        preferred_height: Preferred Z coordinate
        avoid_center: Avoid the center of the scene
        
    Returns:
        Position [x, y, z] of empty space
    """
    occupied_spaces = []
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            occupied_spaces.append({
                "center": list(obj.location),
                "radius": max(obj.dimensions) / 2 if any(obj.dimensions) else 0.5
            })
    
    # Cerca spazio libero in una griglia
    for x in [-3, -1.5, 0, 1.5, 3]:
        for y in [-3, -1.5, 0, 1.5, 3]:
            if avoid_center and x == 0 and y == 0:
                continue
                
            position = [x, y, preferred_height]
            
            # Controlla se è libero
            is_free = True
            for occupied in occupied_spaces:
                distance = mathutils.Vector(position).length
                if distance < (size_needed + occupied["radius"]):
                    is_free = False
                    break
            
            if is_free:
                return position
    
    # Se non trova spazio, ritorna posizione offset
    return [len(occupied_spaces) * 2, 0, preferred_height]

def align_objects_in_grid(
    object_names: List[str],
    spacing: float = 2.0,
    columns: int = 3,
    start_position: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Align objects in a grid pattern.
    
    Args:
        object_names: List of object names to align
        spacing: Distance between objects
        columns: Number of columns in grid
        start_position: Starting position for grid
        
    Returns:
        New positions for all objects
    """
    start = start_position or [0, 0, 0]
    positions = {}
    
    for i, obj_name in enumerate(object_names):
        obj = bpy.data.objects.get(obj_name)
        if obj:
            row = i // columns
            col = i % columns
            
            new_position = [
                start[0] + col * spacing,
                start[1] - row * spacing,
                start[2]
            ]
            
            obj.location = new_position
            positions[obj_name] = new_position
    
    return {
        "aligned_objects": positions,
        "grid_size": [columns, (len(object_names) - 1) // columns + 1],
        "spacing": spacing
    }

# ============================================
# 🖼️ RENDERING ENGINE
# ============================================

@thread_safe
def configure_render_settings(
    engine: str = "CYCLES",
    device: str = "GPU",
    samples: int = 128,
    resolution_x: int = 1920,
    resolution_y: int = 1080,
    resolution_percentage: int = 100,
    file_format: str = "PNG",
    color_mode: str = "RGBA",
    color_depth: str = "16",
    compression: int = 15,
    transparent: bool = False,
    use_denoising: bool = True,
    use_motion_blur: bool = False,
    motion_blur_shutter: float = 0.5,
    use_bloom: bool = False,
    bloom_intensity: float = 0.5,
    bloom_threshold: float = 0.8,
    exposure: float = 1.0,
    gamma: float = 1.0,
    use_compositing: bool = True,
    use_sequencer: bool = False
) -> Dict[str, Any]:
    """
    Configure professional render settings.
    
    Args:
        engine: Render engine (CYCLES, EEVEE, WORKBENCH)
        device: Compute device (CPU, GPU)
        samples: Number of samples
        resolution_x: Horizontal resolution
        resolution_y: Vertical resolution
        resolution_percentage: Resolution scale percentage
        file_format: Output format (PNG, JPEG, EXR, TIFF, etc.)
        color_mode: Color mode (RGB, RGBA, BW)
        color_depth: Bit depth (8, 16, 32)
        compression: Compression level (0-100)
        transparent: Transparent background
        use_denoising: Enable denoising
        use_motion_blur: Enable motion blur
        motion_blur_shutter: Motion blur shutter time
        use_bloom: Enable bloom effect (EEVEE)
        bloom_intensity: Bloom intensity
        bloom_threshold: Bloom threshold
        exposure: Exposure adjustment
        gamma: Gamma correction
        use_compositing: Use compositor
        use_sequencer: Use sequencer
        
    Returns:
        Complete render configuration
    """
    scene = bpy.context.scene
    render = scene.render
    
    # Basic settings
    render.engine = engine
    render.resolution_x = resolution_x
    render.resolution_y = resolution_y
    render.resolution_percentage = resolution_percentage
    render.use_compositing = use_compositing
    render.use_sequencer = use_sequencer
    
    # File output settings
    render.image_settings.file_format = file_format
    render.image_settings.color_mode = color_mode
    render.image_settings.color_depth = color_depth
    if file_format in ['PNG', 'TIFF']:
        render.image_settings.compression = compression
    
    # Transparency
    render.film_transparent = transparent
    
    # Color management
    scene.view_settings.exposure = exposure
    scene.view_settings.gamma = gamma
    
    # Engine-specific settings
    if engine == "CYCLES":
        cycles = scene.cycles
        
        # Device settings
        cycles.device = device
        if device == "GPU":
            # Enable GPU compute
            prefs = bpy.context.preferences.addons.get('cycles')
            if prefs:
                compute_device_type = prefs.preferences.compute_device_type
                prefs.preferences.compute_device_type = 'CUDA'  # or 'OPTIX', 'METAL', 'HIP'
        
        # Sampling
        cycles.samples = samples
        cycles.use_adaptive_sampling = True
        cycles.adaptive_threshold = 0.01
        
        # Denoising
        cycles.use_denoising = use_denoising
        if use_denoising:
            cycles.denoiser = 'OPENIMAGEDENOISE'
            cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
        
        # Motion blur
        render.use_motion_blur = use_motion_blur
        if use_motion_blur:
            cycles.motion_blur_position = 'CENTER'
            render.motion_blur_shutter = motion_blur_shutter
            
    elif engine == "EEVEE":
        eevee = scene.eevee
        
        # Sampling
        eevee.taa_render_samples = samples
        eevee.taa_samples = min(16, samples)
        
        # Effects
        eevee.use_bloom = use_bloom
        if use_bloom:
            eevee.bloom_intensity = bloom_intensity
            eevee.bloom_threshold = bloom_threshold
        
        # Motion blur
        eevee.use_motion_blur = use_motion_blur
        if use_motion_blur:
            eevee.motion_blur_shutter = motion_blur_shutter
            eevee.motion_blur_samples = 8
    
    return {
        "engine": engine,
        "device": device if engine == "CYCLES" else "CPU",
        "resolution": f"{resolution_x}x{resolution_y}",
        "resolution_percentage": resolution_percentage,
        "samples": samples,
        "file_format": file_format,
        "color": {
            "mode": color_mode,
            "depth": color_depth,
            "exposure": exposure,
            "gamma": gamma
        },
        "effects": {
            "transparent": transparent,
            "denoising": use_denoising and engine == "CYCLES",
            "motion_blur": use_motion_blur,
            "bloom": use_bloom and engine == "EEVEE"
        },
        "performance": {
            "use_compositing": use_compositing,
            "use_sequencer": use_sequencer
        },
        "configured_at": datetime.now().isoformat()
    }

@thread_safe
def render_image(
    output_path: Optional[str] = None,
    animation: bool = False,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    frame_step: int = 1,
    return_base64: bool = False,
    use_viewport: bool = False
) -> Dict[str, Any]:
    """
    Execute professional rendering.
    
    Args:
        output_path: Output file path
        animation: Render animation
        frame_start: Animation start frame
        frame_end: Animation end frame
        frame_step: Frame step for animation
        return_base64: Return image as base64
        use_viewport: Use viewport render (faster, lower quality)
        
    Returns:
        Complete render result
    """
    scene = bpy.context.scene
    render_start_time = time.time()
    
    # Set frame range for animation
    if animation:
        if frame_start is not None:
            scene.frame_start = frame_start
        if frame_end is not None:
            scene.frame_end = frame_end
        scene.frame_step = frame_step
    
    # Setup output path
    if output_path:
        render.render.filepath = output_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        # Use temporary file
        tmp = tempfile.NamedTemporaryFile(
            suffix='.png' if not animation else '.mp4',
            delete=False
        )
        scene.render.filepath = tmp.name
        output_path = tmp.name
    
    # Execute render
    try:
        if use_viewport:
            # Quick viewport render
            bpy.ops.render.opengl(
                animation=animation,
                write_still=not animation
            )
        else:
            # Full render
            bpy.ops.render.render(
                animation=animation,
                write_still=not animation
            )
    except Exception as e:
        raise RuntimeError(f"Render failed: {e}")
    
    render_time = time.time() - render_start_time
    
    # Get file info
    file_size = 0
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
    
    result = {
        "output_path": output_path,
        "animation": animation,
        "frames_rendered": scene.frame_end - scene.frame_start + 1 if animation else 1,
        "resolution": f"{scene.render.resolution_x}x{scene.render.resolution_y}",
        "engine": scene.render.engine,
        "render_time": render_time,
        "file_size": file_size,
        "viewport_render": use_viewport,
        "rendered_at": datetime.now().isoformat()
    }
    
    # Return base64 if requested (only for still images)
    if return_base64 and not animation and os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            result["image_base64"] = base64.b64encode(f.read()).decode('utf-8')
        
        # Cleanup temp file if created
        if 'tmp' in locals():
            os.unlink(output_path)
    
    return result

# ============================================
# 🎯 SCENE MANAGEMENT (continued)
# ============================================

def get_scene_info() -> Dict[str, Any]:
    """
    Get comprehensive scene information.
    This function doesn't need thread safety as it only reads data.
    
    Returns:
        Complete scene analysis
    """
    scene = bpy.context.scene
    
    # Object analysis
    object_types = {}
    total_verts = 0
    total_edges = 0
    total_faces = 0
    total_tris = 0
    
    for obj in scene.objects:
        obj_type = obj.type
        object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        if hasattr(obj.data, 'vertices'):
            total_verts += len(obj.data.vertices)
        if hasattr(obj.data, 'edges'):
            total_edges += len(obj.data.edges)
        if hasattr(obj.data, 'polygons'):
            total_faces += len(obj.data.polygons)
            # Calculate triangles
            for poly in obj.data.polygons:
                total_tris += len(poly.vertices) - 2
    
    # Memory usage estimation
    memory_usage = {
        "objects": len(bpy.data.objects) * 1024,  # Rough estimate
        "meshes": sum(len(m.vertices) * 32 + len(m.polygons) * 64 for m in bpy.data.meshes),
        "materials": len(bpy.data.materials) * 2048,
        "textures": sum(img.size[0] * img.size[1] * 4 for img in bpy.data.images if img.size[0] > 0)
    }
    memory_usage["total"] = sum(memory_usage.values())
    
    # Collections info
    collections = {}
    for coll in bpy.data.collections:
        collections[coll.name] = {
            "objects": len(coll.objects),
            "children": len(coll.children)
        }
    
    return {
        "scene_name": scene.name,
        "file_path": bpy.data.filepath or "Unsaved",
        "objects": {
            "total": len(scene.objects),
            "visible": len([o for o in scene.objects if o.visible_get()]),
            "selected": len([o for o in scene.objects if o.select_get()]),
            "by_type": object_types,
            "names": [obj.name for obj in scene.objects]
        },
        "geometry": {
            "total_vertices": total_verts,
            "total_edges": total_edges,
            "total_faces": total_faces,
            "total_triangles": total_tris
        },
        "materials": {
            "total": len(bpy.data.materials),
            "used": len([m for m in bpy.data.materials if m.users > 0]),
            "names": [mat.name for mat in bpy.data.materials]
        },
        "textures": {
            "total": len(bpy.data.images),
            "names": [img.name for img in bpy.data.images],
            "total_pixels": sum(img.size[0] * img.size[1] for img in bpy.data.images if img.size[0] > 0)
        },
        "animation": {
            "fps": scene.render.fps,
            "frame_current": scene.frame_current,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "total_frames": scene.frame_end - scene.frame_start + 1,
            "actions": len(bpy.data.actions),
            "armatures": len([o for o in scene.objects if o.type == 'ARMATURE'])
        },
        "render": {
            "engine": scene.render.engine,
            "resolution": f"{scene.render.resolution_x}x{scene.render.resolution_y}",
            "resolution_percentage": scene.render.resolution_percentage,
            "samples": scene.cycles.samples if scene.render.engine == 'CYCLES' else scene.eevee.taa_render_samples,
            "file_format": scene.render.image_settings.file_format
        },
        "cameras": [obj.name for obj in scene.objects if obj.type == 'CAMERA'],
        "lights": [obj.name for obj in scene.objects if obj.type == 'LIGHT'],
        "active_camera": scene.camera.name if scene.camera else None,
        "collections": collections,
        "memory_usage_bytes": memory_usage,
        "statistics": {
            "modifiers": sum(len(o.modifiers) for o in scene.objects),
            "constraints": sum(len(o.constraints) for o in scene.objects),
            "shape_keys": sum(1 for o in scene.objects if hasattr(o.data, 'shape_keys') and o.data.shape_keys)
        },
        "queried_at": datetime.now().isoformat()
    }

@thread_safe
def clear_scene(
    clear_objects: bool = True,
    clear_materials: bool = True,
    clear_textures: bool = True,
    clear_animations: bool = False,
    clear_node_groups: bool = False,
    clear_worlds: bool = False,
    keep_cameras: bool = True,
    keep_lights: bool = True
) -> Dict[str, Any]:
    """
    Clear scene with granular control.
    
    Args:
        clear_objects: Remove all objects
        clear_materials: Remove all materials
        clear_textures: Remove all textures/images
        clear_animations: Remove all animations
        clear_node_groups: Remove node groups
        clear_worlds: Remove world settings
        keep_cameras: Preserve cameras
        keep_lights: Preserve lights
        
    Returns:
        Detailed clear operation report
    """
    removed = {
        "objects": 0,
        "materials": 0,
        "textures": 0,
        "animations": 0,
        "node_groups": 0,
        "worlds": 0,
        "meshes": 0,
        "curves": 0
    }
    
    preserved = {
        "cameras": [],
        "lights": []
    }
    
    if clear_objects:
        # Preserve cameras and lights if requested
        for obj in list(bpy.data.objects):
            should_remove = True
            
            if keep_cameras and obj.type == 'CAMERA':
                preserved["cameras"].append(obj.name)
                should_remove = False
            elif keep_lights and obj.type == 'LIGHT':
                preserved["lights"].append(obj.name)
                should_remove = False
            
            if should_remove:
                bpy.data.objects.remove(obj)
                removed["objects"] += 1
    
    # Clean orphaned data
    if clear_materials:
        for mat in list(bpy.data.materials):
            if mat.users == 0 or clear_materials:
                bpy.data.materials.remove(mat)
                removed["materials"] += 1
    
    if clear_textures:
        for img in list(bpy.data.images):
            if img.users == 0 or clear_textures:
                bpy.data.images.remove(img)
                removed["textures"] += 1
    
    if clear_animations:
        for action in list(bpy.data.actions):
            bpy.data.actions.remove(action)
            removed["animations"] += 1
    
    if clear_node_groups:
        for ng in list(bpy.data.node_groups):
            bpy.data.node_groups.remove(ng)
            removed["node_groups"] += 1
    
    if clear_worlds:
        for world in list(bpy.data.worlds):
            if world != bpy.context.scene.world:
                bpy.data.worlds.remove(world)
                removed["worlds"] += 1
    
    # Clean orphaned mesh data
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
            removed["meshes"] += 1
    
    # Clean orphaned curve data
    for curve in list(bpy.data.curves):
        if curve.users == 0:
            bpy.data.curves.remove(curve)
            removed["curves"] += 1
    
    return {
        "removed": removed,
        "preserved": preserved,
        "total_removed": sum(removed.values()),
        "cleared_at": datetime.now().isoformat()
    }

# ============================================
# 📁 FILE OPERATIONS
# ============================================

@thread_safe
def save_blend_file(
    file_path: str,
    compress: bool = False,
    copy: bool = False,
    auto_pack: bool = False,
    save_preview: bool = True
) -> Dict[str, Any]:
    """
    Save Blender file with options.
    
    Args:
        file_path: Output path for .blend file
        compress: Compress file
        copy: Save a copy (don't change current filepath)
        auto_pack: Pack external data into blend file
        save_preview: Generate file preview
        
    Returns:
        Complete save information
    """
    # Ensure .blend extension
    if not file_path.endswith('.blend'):
        file_path += '.blend'
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Pack external data if requested
    if auto_pack:
        bpy.ops.file.pack_all()
    
    # Store original path if copying
    original_path = bpy.data.filepath if copy else None
    
    try:
        if copy:
            bpy.ops.wm.save_as_mainfile(
                filepath=file_path,
                compress=compress,
                copy=True
            )
        else:
            bpy.ops.wm.save_mainfile(
                filepath=file_path,
                compress=compress
            )
    except Exception as e:
        raise RuntimeError(f"Failed to save file: {e}")
    
    # Get file statistics
    file_stats = os.stat(file_path)
    
    return {
        "file_path": file_path,
        "file_size": file_stats.st_size,
        "file_size_mb": file_stats.st_size / (1024 * 1024),
        "compressed": compress,
        "copy": copy,
        "original_path": original_path,
        "auto_packed": auto_pack,
        "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
        "saved_at": datetime.now().isoformat()
    }

@thread_safe
def import_file(
    file_path: str,
    file_format: Optional[str] = None,
    use_selection: bool = False,
    axis_forward: str = '-Z',
    axis_up: str = 'Y',
    scale: float = 1.0,
    apply_transform: bool = True
) -> Dict[str, Any]:
    """
    Import 3D files with comprehensive options.
    
    Args:
        file_path: Path to file
        file_format: Format (auto-detect if None)
        use_selection: Import as selection only
        axis_forward: Forward axis mapping
        axis_up: Up axis mapping
        scale: Import scale
        apply_transform: Apply transformation after import
        
    Returns:
        Complete import information
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    # Auto-detect format
    if file_format is None:
        ext = os.path.splitext(file_path)[1].lower()
        file_format = ext[1:]  # Remove dot
    
    # Store objects before import
    before = set(bpy.data.objects.keys())
    import_start_time = time.time()
    
    # Import based on format
    import_functions = {
        'fbx': lambda: bpy.ops.import_scene.fbx(
            filepath=file_path,
            axis_forward=axis_forward,
            axis_up=axis_up,
            global_scale=scale,
            use_manual_orientation=True
        ),
        'obj': lambda: bpy.ops.import_scene.obj(
            filepath=file_path,
            axis_forward=axis_forward,
            axis_up=axis_up,
            global_scale=scale
        ),
        'gltf': lambda: bpy.ops.import_scene.gltf(
            filepath=file_path
        ),
        'glb': lambda: bpy.ops.import_scene.gltf(
            filepath=file_path
        ),
        'dae': lambda: bpy.ops.wm.collada_import(
            filepath=file_path
        ),
        'stl': lambda: bpy.ops.import_mesh.stl(
            filepath=file_path,
            global_scale=scale
        ),
        'ply': lambda: bpy.ops.import_mesh.ply(
            filepath=file_path
        ),
        'abc': lambda: bpy.ops.wm.alembic_import(
            filepath=file_path,
            scale=scale
        ),
        'usd': lambda: bpy.ops.wm.usd_import(
            filepath=file_path,
            scale=scale
        )
    }
    
    file_format = file_format.lower()
    if file_format not in import_functions:
        raise ValueError(f"Unsupported format: {file_format}")
    
    # Execute import
    try:
        import_functions[file_format]()
    except Exception as e:
        raise RuntimeError(f"Import failed: {e}")
    
    import_time = time.time() - import_start_time
    
    # Get imported objects
    after = set(bpy.data.objects.keys())
    imported_objects = list(after - before)
    
    # Apply transform if requested
    if apply_transform and imported_objects:
        for obj_name in imported_objects:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Calculate statistics
    total_verts = 0
    total_faces = 0
    for obj_name in imported_objects:
        obj = bpy.data.objects.get(obj_name)
        if obj and hasattr(obj.data, 'vertices'):
            total_verts += len(obj.data.vertices)
        if obj and hasattr(obj.data, 'polygons'):
            total_faces += len(obj.data.polygons)
    
    file_stats = os.stat(file_path)
    
    return {
        "file_path": file_path,
        "format": file_format,
        "imported_objects": imported_objects,
        "object_count": len(imported_objects),
        "statistics": {
            "total_vertices": total_verts,
            "total_faces": total_faces
        },
        "import_settings": {
            "axis_forward": axis_forward,
            "axis_up": axis_up,
            "scale": scale,
            "transform_applied": apply_transform
        },
        "file_size": file_stats.st_size,
        "import_time": import_time,
        "imported_at": datetime.now().isoformat()
    }

@thread_safe
def export_file(
    file_path: str,
    file_format: str,
    selected_only: bool = False,
    apply_modifiers: bool = True,
    axis_forward: str = '-Z',
    axis_up: str = 'Y',
    scale: float = 1.0,
    use_mesh_modifiers: bool = True,
    use_metadata: bool = True
) -> Dict[str, Any]:
    """
    Export with professional options.
    
    Args:
        file_path: Output file path
        file_format: Export format
        selected_only: Export only selected objects
        apply_modifiers: Apply modifiers before export
        axis_forward: Forward axis for export
        axis_up: Up axis for export
        scale: Export scale
        use_mesh_modifiers: Apply mesh modifiers
        use_metadata: Include metadata
        
    Returns:
        Complete export information
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Count objects to export
    if selected_only:
        objects_to_export = [o for o in bpy.data.objects if o.select_get()]
    else:
        objects_to_export = list(bpy.data.objects)
    
    export_start_time = time.time()
    
    # Export based on format
    export_functions = {
        'fbx': lambda: bpy.ops.export_scene.fbx(
            filepath=file_path,
            use_selection=selected_only,
            use_mesh_modifiers=use_mesh_modifiers,
            axis_forward=axis_forward,
            axis_up=axis_up,
            global_scale=scale,
            use_metadata=use_metadata
        ),
        'obj': lambda: bpy.ops.export_scene.obj(
            filepath=file_path,
            use_selection=selected_only,
            use_mesh_modifiers=use_mesh_modifiers,
            axis_forward=axis_forward,
            axis_up=axis_up,
            global_scale=scale
        ),
        'gltf': lambda: bpy.ops.export_scene.gltf(
            filepath=file_path,
            use_selection=selected_only,
            export_apply=apply_modifiers
        ),
        'glb': lambda: bpy.ops.export_scene.gltf(
            filepath=file_path,
            use_selection=selected_only,
            export_apply=apply_modifiers,
            export_format='GLB'
        ),
        'dae': lambda: bpy.ops.wm.collada_export(
            filepath=file_path,
            selected=selected_only,
            apply_modifiers=apply_modifiers
        ),
        'stl': lambda: bpy.ops.export_mesh.stl(
            filepath=file_path,
            use_selection=selected_only,
            use_mesh_modifiers=use_mesh_modifiers,
            global_scale=scale
        ),
        'ply': lambda: bpy.ops.export_mesh.ply(
            filepath=file_path,
            use_selection=selected_only,
            use_mesh_modifiers=use_mesh_modifiers,
            global_scale=scale
        ),
        'abc': lambda: bpy.ops.wm.alembic_export(
            filepath=file_path,
            selected=selected_only,
            apply_subdiv=apply_modifiers,
            global_scale=scale
        ),
        'usd': lambda: bpy.ops.wm.usd_export(
            filepath=file_path,
            selected_objects_only=selected_only,
            export_materials=True
        )
    }
    
    file_format = file_format.lower()
    if file_format not in export_functions:
        raise ValueError(f"Unsupported export format: {file_format}")
    
    # Execute export
    try:
        export_functions[file_format]()
    except Exception as e:
        raise RuntimeError(f"Export failed: {e}")
    
    export_time = time.time() - export_start_time
    
    # Get file statistics
    file_size = 0
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
    
    return {
        "file_path": file_path,
        "format": file_format,
        "objects_exported": len(objects_to_export),
        "selected_only": selected_only,
        "settings": {
            "apply_modifiers": apply_modifiers,
            "axis_forward": axis_forward,
            "axis_up": axis_up,
            "scale": scale
        },
        "file_size": file_size,
        "file_size_mb": file_size / (1024 * 1024),
        "export_time": export_time,
        "exported_at": datetime.now().isoformat()
    }

# ============================================
# 🔗 ADVANCED OPERATIONS
# ============================================

@thread_safe
def boolean_operation(
    object_a: str,
    object_b: str,
    operation: str = "DIFFERENCE",
    solver: str = "FAST",
    apply: bool = True,
    hide_b: bool = True,
    delete_b: bool = False
) -> Dict[str, Any]:
    """
    Perform boolean operations with advanced options.
    
    Args:
        object_a: First object (target)
        object_b: Second object (operator)
        operation: Operation type (UNION, DIFFERENCE, INTERSECT)
        solver: Solver type (FAST, EXACT)
        apply: Apply modifier immediately
        hide_b: Hide second object
        delete_b: Delete second object after operation
        
    Returns:
        Complete boolean result
    """
    obj_a = bpy.data.objects.get(object_a)
    obj_b = bpy.data.objects.get(object_b)
    
    if not obj_a:
        raise ValueError(f"Object A '{object_a}' not found")
    if not obj_b:
        raise ValueError(f"Object B '{object_b}' not found")
    
    # Store initial statistics
    stats_before = {
        "vertices": len(obj_a.data.vertices) if hasattr(obj_a.data, 'vertices') else 0,
        "faces": len(obj_a.data.polygons) if hasattr(obj_a.data, 'polygons') else 0
    }
    
    # Add boolean modifier
    modifier = obj_a.modifiers.new(name=f"Boolean_{operation}", type='BOOLEAN')
    modifier.operation = operation
    modifier.object = obj_b
    modifier.solver = solver
    
    # Apply if requested
    if apply:
        bpy.context.view_layer.objects.active = obj_a
        obj_a.select_set(True)
        try:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        except Exception as e:
            raise RuntimeError(f"Boolean operation failed: {e}")
    
    # Handle object B
    if delete_b:
        bpy.data.objects.remove(obj_b)
    elif hide_b:
        obj_b.hide_set(True)
        obj_b.hide_render = True
    
    # Get final statistics
    stats_after = {
        "vertices": len(obj_a.data.vertices) if hasattr(obj_a.data, 'vertices') else 0,
        "faces": len(obj_a.data.polygons) if hasattr(obj_a.data, 'polygons') else 0
    }
    
    return {
        "object_a": object_a,
        "object_b": object_b if not delete_b else "deleted",
        "operation": operation,
        "solver": solver,
        "applied": apply,
        "statistics": {
            "before": stats_before,
            "after": stats_after,
            "difference": {
                "vertices": stats_after["vertices"] - stats_before["vertices"],
                "faces": stats_after["faces"] - stats_before["faces"]
            }
        },
        "object_b_status": "deleted" if delete_b else ("hidden" if hide_b else "visible"),
        "completed_at": datetime.now().isoformat()
    }

# ============================================
# 🚀 SERVER MANAGEMENT
# ============================================

def create_blender_mcp_server():
    """
    Create production-ready Blender MCP server with all functions.
    
    Returns:
        FastAPI app configured for complete Blender control
    """
    # Complete list of all production tools
    all_tools = [

        # SPATIAL AWARENESS TOOLS
        get_object_position,
        calculate_position_relative_to,
        get_all_objects_positions,
        find_empty_space,
        align_objects_in_grid,

        # Object Creation
        create_mesh_object,
        create_curve_object,
        create_text_object,
        
        # Object Manipulation
        transform_object,
        duplicate_object,
        delete_objects,
        
        # Modifiers
        add_modifier,
        apply_modifier,
        
        # Materials & Shading
        create_material,
        assign_material,
        
        # Lighting
        create_light,
        
        # Camera
        create_camera,
        
        # Animation
        create_keyframe,
        
        # Rendering
        configure_render_settings,
        render_image,
        
        # File Operations
        import_file,
        export_file,
        save_blend_file,
        
        # Scene Management
        get_scene_info,
        clear_scene,
        
        # Advanced Operations
        boolean_operation,

                create_particle_system,
        add_force_field,
        
        # Geometry Nodes
        add_geometry_nodes,
        create_procedural_geometry,
        
        # UV & Texturing
        unwrap_uv,
        add_texture_paint_slots,
        
        # Batch Operations
        batch_create_objects,
        batch_transform,
        
        # Simulations
        setup_rigid_body,
        add_cloth_simulation,
        setup_fluid_simulation,
        add_fluid_flow,
        
        # Advanced Nodes
        create_shader_node_tree,
        create_procedural_material,
        
        # Templates
        create_from_template,
        
        # Grease Pencil
        create_grease_pencil_drawing,
        
        # Performance
        optimize_scene,
        clear_cache,
        
        # Presets
        quick_scene_setup,
        create_hdri_environment,

        # Additional Advanced Tools
        set_optimal_camera_for_all,
        auto_arrange_objects,
        verify_last_operation,
        analyze_spatial_layout,
        capture_viewport_image
    ]
    
    # Create MCP server
    app = expose_tools(
        tools=all_tools,
        title="Blender MCP Server - Production",
        description="Production-ready MCP server for complete Blender control",
        version="3.0.0"
    )
    
    logger.info(f"MCP Server created with {len(all_tools)} production tools")
    
    return app

# ============================================
# 🎮 BLENDER ADDON INTERFACE
# ============================================

# Global server instance
server_thread = None
server_app = None

class MCPSERVER_PT_main_panel(bpy.types.Panel):
    """Main MCP Server Panel"""
    bl_label = "MCP Server Control"
    bl_idname = "MCPSERVER_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MCP Server"
    
    def draw(self, context):
        layout = self.layout
        
        # Server status
        box = layout.box()
        box.label(text="Server Status", icon='WORLD_DATA')
        
        row = box.row(align=True)
        if server_thread is None or not server_thread.is_alive():
            row.operator("mcp.start_server", text="Start Server", icon='PLAY')
        else:
            row.operator("mcp.stop_server", text="Stop Server", icon='PAUSE')
            
            col = box.column()
            col.label(text="Server running on:", icon='URL')
            col.label(text=f"http://localhost:{Config.PORT}")
            
            row = col.row()
            row.operator("wm.url_open", text="API Docs").url = f"http://localhost:{Config.PORT}/docs"
            row.operator("wm.url_open", text="List Tools").url = f"http://localhost:{Config.PORT}/mcp/list_tools"
        
        # Server info
        box = layout.box()
        box.label(text="Server Information", icon='INFO')
        col = box.column(align=True)
        col.label(text=f"Version: 3.0.0")
        col.label(text=f"Thread-Safe: {'Yes' if Config.THREAD_SAFE_OPERATIONS else 'No'}")
        col.label(text=f"Port: {Config.PORT}")
        
        # Quick stats
        box = layout.box()
        box.label(text="Scene Statistics", icon='SCENE_DATA')
        col = box.column(align=True)
        col.label(text=f"Objects: {len(bpy.data.objects)}")
        col.label(text=f"Materials: {len(bpy.data.materials)}")
        col.label(text=f"Textures: {len(bpy.data.images)}")
        col.label(text=f"Frame: {context.scene.frame_current}/{context.scene.frame_end}")

class MCPSERVER_OT_start_server(bpy.types.Operator):
    """Start MCP Server"""
    bl_idname = "mcp.start_server"
    bl_label = "Start MCP Server"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        global server_thread, server_app
        
        try:
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()
            
            # Avvia il processore di code
            start_queue_processor()
            
            # Create MCP server app
            server_app = create_blender_mcp_server()
            
            # Run in separate thread
            def run_server():
                uvicorn.run(server_app, host="0.0.0.0", port=8000)
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            self.report({'INFO'}, "MCP Server started on http://localhost:8000")
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start server: {str(e)}")
            
        return {'FINISHED'}

class MCPSERVER_OT_start_server(bpy.types.Operator):
    """Start MCP Server"""
    bl_idname = "mcp.start_server"
    bl_label = "Start MCP Server"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        global server_thread, server_app
        
        try:
            # Start thread-safe executor
            thread_executor.start()
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()
            # Create MCP server app
            server_app = create_blender_mcp_server()
            
            # Run in separate thread
            def run_server():
                uvicorn.run(
                    server_app,
                    host=Config.HOST,
                    port=Config.PORT,
                    log_level="info"
                )
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            self.report({'INFO'}, f"MCP Server started on http://localhost:{Config.PORT}")
            logger.info(f"MCP Server started successfully on port {Config.PORT}")
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start server: {str(e)}")
            logger.error(f"Failed to start server: {e}")
            logger.exception(e)
            
        return {'FINISHED'}


class MCPSERVER_OT_stop_server(bpy.types.Operator):
    """Stop MCP Server"""
    bl_idname = "mcp.stop_server"
    bl_label = "Stop MCP Server"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        global server_thread
        
        # Stop thread-safe executor
        thread_executor.stop()
        
        # Stop server thread
        server_thread = None
        
        self.report({'INFO'}, "MCP Server stopped")
        logger.info("MCP Server stopped")
        
        return {'FINISHED'}


# Registration
classes = [
    MCPSERVER_PT_main_panel,
    MCPSERVER_OT_start_server,
    MCPSERVER_OT_stop_server,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    logger.info("MCP Server addon registered")

def unregister():
    # Stop server if running
    if server_thread and server_thread.is_alive():
        thread_executor.stop()
    
    for cls in classes:
        bpy.utils.unregister_class(cls)
    logger.info("MCP Server addon unregistered")

if __name__ == "__main__":
    register()