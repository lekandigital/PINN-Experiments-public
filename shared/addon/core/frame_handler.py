"""
Frame Handler - Bridges Blender's animation system with neural simulation backends.

Registers handlers for frame changes and manages simulation state
across timeline navigation (forward, backward, scrubbing).
"""

from typing import Optional
import time

# Global state for frame handling
_is_registered = False
_simulation_state = None
_last_simulated_frame = -1


def register_frame_handler() -> None:
    """Register the frame change handler with Blender."""
    global _is_registered
    
    import bpy
    
    if _is_registered:
        return
    
    if on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(on_frame_change)
    
    _is_registered = True
    print("[NeuralSim] Frame handler registered")


def unregister_frame_handler() -> None:
    """Remove the frame change handler."""
    global _is_registered
    
    import bpy
    
    if on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(on_frame_change)
    
    _is_registered = False
    print("[NeuralSim] Frame handler unregistered")


def reset_simulation_state() -> None:
    """Reset the cached simulation state."""
    global _simulation_state, _last_simulated_frame
    _simulation_state = None
    _last_simulated_frame = -1


def on_frame_change(scene, depsgraph=None) -> None:
    """
    Called by Blender after every frame change (playback, scrubbing, stepping).
    
    Handles:
    - Sequential frame advance (normal playback)
    - Backward scrubbing (reset and re-simulate)
    - Frame skipping (fill missing frames)
    - Stateless vs stateful models
    """
    global _simulation_state, _last_simulated_frame
    
    from ..backend import get_manager, SimulationState
    from .request_builder import build_request
    from .result_applier import apply_result
    
    # Check if addon properties exist
    if not hasattr(scene, "neural_sim"):
        return
    
    props = scene.neural_sim
    
    # Check if simulation is active
    if not props.is_loaded or not props.is_playing:
        return
    
    manager = get_manager()
    caps = manager.get_active_capabilities()
    
    if caps is None:
        return
    
    current_frame = scene.frame_current
    fps = scene.render.fps
    current_time = current_frame / fps
    
    # Handle timeline navigation for sequential models
    if caps.needs_sequential_frames:
        _simulation_state = _handle_sequential_model(
            scene, current_frame, _simulation_state, _last_simulated_frame, manager, caps
        )
    else:
        # Stateless model — can jump to any frame directly
        request = build_request(scene, current_frame, None)
        result = manager.predict(request)
        _simulation_state = result.state
        apply_result(scene, result, caps)
    
    # Update tracking
    _last_simulated_frame = current_frame
    
    # Update UI properties
    if hasattr(props, "last_inference_ms"):
        props.last_inference_ms = result.inference_time_ms if 'result' in dir() else 0.0
    if hasattr(props, "current_frame"):
        props.current_frame = current_frame
    if hasattr(props, "current_fps") and result.inference_time_ms > 0:
        props.current_fps = 1000.0 / result.inference_time_ms
    
    # Trigger viewport update
    _refresh_viewports()


def _handle_sequential_model(scene, current_frame, state, last_frame, manager, caps):
    """
    Handle frame changes for models that require sequential execution (GRU-based).
    
    Returns updated simulation state.
    """
    from .request_builder import build_request
    from .result_applier import apply_result
    from ..backend import SimulationState
    
    props = scene.neural_sim
    
    # Case 1: First frame or reset
    if last_frame < 0 or current_frame == scene.frame_start:
        manager.reset()
        state = None
        request = build_request(scene, current_frame, state)
        result = manager.predict(request)
        apply_result(scene, result, caps)
        return result.state
    
    # Case 2: Backward scrubbing — must reset and re-simulate
    if current_frame < last_frame:
        manager.reset()
        state = None
        
        # Warn if too many frames
        num_frames_to_simulate = current_frame - scene.frame_start + 1
        if num_frames_to_simulate > 100:
            print(f"[NeuralSim] Warning: Re-simulating {num_frames_to_simulate} frames after backward scrub")
        
        # Re-simulate all frames up to current
        for f in range(scene.frame_start, current_frame + 1):
            request = build_request(scene, f, state)
            result = manager.predict(request)
            state = result.state
        
        apply_result(scene, result, caps)
        return state
    
    # Case 3: Frame skip — simulate missing frames
    if current_frame > last_frame + 1:
        for f in range(last_frame + 1, current_frame + 1):
            request = build_request(scene, f, state)
            result = manager.predict(request)
            state = result.state
        
        apply_result(scene, result, caps)
        return state
    
    # Case 4: Normal sequential advance
    request = build_request(scene, current_frame, state)
    result = manager.predict(request)
    apply_result(scene, result, caps)
    
    # Check stability warning
    if hasattr(props, "current_frame"):
        if current_frame > caps.max_stable_frames * 0.8:
            print(f"[NeuralSim] Warning: Approaching rollout stability limit ({caps.max_stable_frames} frames)")
    
    return result.state


def _refresh_viewports() -> None:
    """Trigger viewport refresh for all 3D views."""
    import bpy
    
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def step_simulation(scene, direction: int = 1) -> None:
    """
    Step the simulation forward or backward.
    
    Args:
        scene: Blender scene
        direction: +1 for forward, -1 for backward
    """
    scene.frame_set(scene.frame_current + direction)


def jump_to_frame(scene, frame: int) -> None:
    """
    Jump directly to a specific frame.
    
    Args:
        scene: Blender scene
        frame: Target frame number
    """
    scene.frame_set(frame)


def get_simulation_state():
    """Get the current simulation state (for external access)."""
    global _simulation_state
    return _simulation_state


def set_simulation_state(state) -> None:
    """Set the simulation state (for state restoration)."""
    global _simulation_state
    _simulation_state = state
