import random

def generate_description(metadata):
    """
    Generate a physics description for the simulation based on metadata.
    
    Args:
        metadata (dict): Dictionary containing 'reynolds_number' and optionally 'prompt'.
        
    Returns:
        str: A text description.
    """
    re = metadata.get('reynolds_number', 0.0)
    prompt = metadata.get('prompt', "")
    
    # Clean prompt if it's byte string
    if isinstance(prompt, bytes):
        prompt = prompt.decode('utf-8')
    
    # Physics classification
    flow_regime = "laminar" if re < 47 else "unsteady laminar" if re < 200 else "turbulent transition"
    phenomena = []
    
    if re > 47:
        phenomena.append("vortex shedding")
        phenomena.append("Von Karman vortex street")
        
    if re > 150:
        phenomena.append("chaotic wake patterns")
        
    # Templates
    templates = [
        f"A 2D simulation of fluid flow over a cylinder at Reynolds number {re:.1f}.",
        f"Fluid dynamics simulation showing {flow_regime} flow around a circular obstacle (Re={re:.1f}).",
        f" Visualization of {flow_regime} flow conditions at Re={re:.1f}.",
    ]
    
    base_desc = random.choice(templates)
    
    # Add phenomena detail
    if phenomena:
        detail = f" The flow exhibits {', '.join(phenomena)}."
        base_desc += detail
        
    # Add original prompt context (Scenario details)
    if prompt and len(prompt) > 5:
        base_desc += f" Scenario: {prompt}"
             
    return base_desc

if __name__ == "__main__":
    # Test
    print(generate_description({'reynolds_number': 30.0}))
    print(generate_description({'reynolds_number': 100.0}))
    print(generate_description({'reynolds_number': 250.0}))
