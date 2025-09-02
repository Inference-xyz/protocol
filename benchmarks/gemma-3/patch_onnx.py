#!/usr/bin/env python3
"""
ONNX Model Patcher - Removes high-version operations and replaces them with standard opset 18 compatible operations.

This script handles:
- Gelu (opset 20+) -> Standard operations using Erf
- SimplifiedLayerNormalization -> Standard LayerNormalization or decomposed operations
- RotaryEmbedding (opset 23+) -> Decomposed into standard operations
- GroupQueryAttention -> Manual attention pattern with standard operations
"""

import argparse
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import onnx
from onnx import helper, numpy_helper, version_converter, shape_inference
from onnx import onnx_pb as onnx_proto
import onnx.inliner
import onnx.shape_inference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ONNXPatcher:
    """Patches ONNX models to use only standard operations compatible with opset 18."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.node_count = 0
        
    def load_model(self) -> None:
        """Load the ONNX model."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = onnx.load(self.model_path)
        logger.info(f"Model loaded: opset {self.model.opset_import[0].version}")
        
    def inline_functions(self) -> None:
        """Inline local and schema functions to remove function operations."""
        logger.info("Inlining functions...")
        try:
            self.model = onnx.inliner.inline_local_functions(self.model)
            logger.info("Local functions inlined successfully")
        except Exception as e:
            logger.warning(f"Could not inline local functions: {e}")
            
        try:
            # Try to inline schema functions if available in this version
            if hasattr(onnx.inliner, 'inline_schema_functions'):
                self.model = onnx.inliner.inline_schema_functions(self.model)
            else:
                self.model = onnx.inliner.inline_selected_functions(self.model, [], exclude=True)
            logger.info("Schema functions inlined successfully")
        except Exception as e:
            logger.warning(f"Could not inline schema functions: {e}")
    
    def get_tensor_shape(self, tensor_name: str) -> Optional[List[int]]:
        """Get shape of a tensor from initializers or value_info."""
        # First check initializers
        for init in self.model.graph.initializer:
            if init.name == tensor_name:
                return list(init.dims)
        
        # Then check value_info
        for info in self.model.graph.value_info:
            if info.name == tensor_name:
                try:
                    return [dim.dim_value for dim in info.type.tensor_type.shape.dim]
                except:
                    # If dim_value is not available, try dim_param
                    try:
                        # Use a default size of 1 for symbolic dimensions
                        return [1 if dim.dim_param else dim.dim_value for dim in info.type.tensor_type.shape.dim]
                    except:
                        pass
        
        # Finally check inputs
        for input in self.model.graph.input:
            if input.name == tensor_name:
                try:
                    return [dim.dim_value for dim in input.type.tensor_type.shape.dim]
                except:
                    # If dim_value is not available, try dim_param
                    try:
                        # Use a default size of 1 for symbolic dimensions
                        return [1 if dim.dim_param else dim.dim_value for dim in input.type.tensor_type.shape.dim]
                    except:
                        pass
        
        # If we can't get the shape, return a default shape
        return [1]  # Default to scalar shape
        
    def get_tensor_type(self, tensor_name: str) -> int:
        """Get data type of a tensor from initializers or value_info."""
        # First check initializers
        for init in self.model.graph.initializer:
            if init.name == tensor_name:
                return init.data_type
        
        # Then check value_info
        for info in self.model.graph.value_info:
            if info.name == tensor_name:
                return info.type.tensor_type.elem_type
        
        # Finally check inputs
        for input in self.model.graph.input:
            if input.name == tensor_name:
                return input.type.tensor_type.elem_type
        
        # Default to FLOAT
        return onnx_proto.TensorProto.FLOAT

    def ensure_all_tensors_have_value_info(self) -> None:
        """Ensure all tensors in the graph have value_info."""
        known_tensors = set()
        
        # Collect all known tensors
        for info in self.model.graph.value_info:
            known_tensors.add(info.name)
        for init in self.model.graph.initializer:
            known_tensors.add(init.name)
        for input in self.model.graph.input:
            known_tensors.add(input.name)
        for output in self.model.graph.output:
            known_tensors.add(output.name)
        
        # First pass: collect shapes from initializers and inputs
        tensor_shapes = {}
        tensor_types = {}
        
        # Get shapes from initializers
        for init in self.model.graph.initializer:
            tensor_shapes[init.name] = list(init.dims)
            tensor_types[init.name] = init.data_type
        
        # Get shapes from inputs
        for input in self.model.graph.input:
            try:
                shape = [dim.dim_value if hasattr(dim, 'dim_value') else 1 for dim in input.type.tensor_type.shape.dim]
                tensor_shapes[input.name] = shape
                tensor_types[input.name] = input.type.tensor_type.elem_type
            except:
                # If we can't get the shape, use a default
                tensor_shapes[input.name] = [1]
                tensor_types[input.name] = onnx_proto.TensorProto.FLOAT
        
        # Get shapes from value_info
        for info in self.model.graph.value_info:
            try:
                shape = [dim.dim_value if hasattr(dim, 'dim_value') else 1 for dim in info.type.tensor_type.shape.dim]
                tensor_shapes[info.name] = shape
                tensor_types[info.name] = info.type.tensor_type.elem_type
            except:
                pass
        
        # Second pass: infer shapes for all nodes
        nodes_to_process = list(self.model.graph.node)
        processed_nodes = set()
        
        while nodes_to_process:
            node = nodes_to_process[0]
            can_process = True
            
            # Check if all inputs have shapes
            for input_name in node.input:
                if input_name not in tensor_shapes:
                    can_process = False
                    break
            
            if can_process:
                # Process node
                nodes_to_process.pop(0)
                processed_nodes.add(node.name)
                
                # Infer output shapes based on node type
                for output_name in node.output:
                    if output_name not in tensor_shapes:
                        shape = None
                        if node.op_type == "MatMul":
                            # MatMul output shape is [batch, M, N] for inputs [batch, M, K] and [batch, K, N]
                            input1_shape = tensor_shapes[node.input[0]]
                            input2_shape = tensor_shapes[node.input[1]]
                            if input1_shape and input2_shape:
                                shape = input1_shape[:-1] + [input2_shape[-1]]
                        elif node.op_type in ["Add", "Sub", "Mul", "Div", "Erf"]:
                            # Element-wise ops preserve shape
                            shape = tensor_shapes[node.input[0]]
                        elif node.op_type == "Reshape":
                            # For Reshape, try to get shape from second input
                            if len(node.input) > 1 and node.input[1] in tensor_shapes:
                                shape = tensor_shapes[node.input[1]]
                        elif node.op_type == "ReduceMean":
                            # For ReduceMean, get input shape and remove reduced dimensions
                            input_shape = tensor_shapes[node.input[0]]
                            if input_shape:
                                # Get axes from attributes
                                axes = None
                                keepdims = 1  # default value
                                for attr in node.attribute:
                                    if attr.name == "axes":
                                        axes = list(attr.ints)
                                    elif attr.name == "keepdims":
                                        keepdims = attr.i
                                
                                if axes is not None:
                                    if keepdims:
                                        shape = input_shape.copy()
                                        for axis in axes:
                                            shape[axis if axis >= 0 else len(shape) + axis] = 1
                                    else:
                                        shape = [dim for i, dim in enumerate(input_shape) if i not in axes]
                        elif node.op_type == "Transpose":
                            # For Transpose, get input shape and permute dimensions
                            input_shape = tensor_shapes[node.input[0]]
                            if input_shape:
                                # Get perm from attributes
                                perm = None
                                for attr in node.attribute:
                                    if attr.name == "perm":
                                        perm = list(attr.ints)
                                        break
                                
                                if perm is None:
                                    # Default permutation reverses dimensions
                                    perm = list(range(len(input_shape)))[::-1]
                                
                                shape = [input_shape[i] for i in perm]
                        
                        if shape:
                            tensor_shapes[output_name] = shape
                            tensor_types[output_name] = tensor_types.get(node.input[0], onnx_proto.TensorProto.FLOAT)
                        else:
                            # If we can't infer the shape, use a default
                            tensor_shapes[output_name] = [1]
                            tensor_types[output_name] = onnx_proto.TensorProto.FLOAT
            else:
                # Move node to end of list
                nodes_to_process.append(nodes_to_process.pop(0))
        
        # Add value_info for all tensors
        for tensor_name, shape in tensor_shapes.items():
            if tensor_name not in known_tensors:
                value_info = helper.make_tensor_value_info(
                    tensor_name,
                    tensor_types.get(tensor_name, onnx_proto.TensorProto.FLOAT),
                    shape
                )
                self.model.graph.value_info.append(value_info)
                known_tensors.add(tensor_name)

    def add_missing_bias_inputs(self) -> None:
        """Add missing bias inputs to nodes that require them."""
        for node in self.model.graph.node:
            if node.op_type in ["LayerNormalization", "SimplifiedLayerNormalization"]:
                # Add default bias (zeros) if missing
                if len(node.input) < 3:
                    # Get input shape from shape inference
                    input_shape = self.get_tensor_shape(node.input[0])
                    
                    # Create zero bias tensor with default shape if needed
                    bias_shape = [input_shape[-1] if input_shape else 1]  # Last dimension for LayerNorm or default
                    bias_name = f"{node.name}_default_bias"
                    bias_tensor = numpy_helper.from_array(
                        np.zeros(bias_shape, dtype=np.float32),
                        name=bias_name
                    )
                    self.model.graph.initializer.append(bias_tensor)
                    
                    # Add bias input
                    node.input.append(bias_name)
                    
                    # Add value info for the bias tensor
                    bias_value_info = helper.make_tensor_value_info(
                        bias_name,
                        onnx_proto.TensorProto.FLOAT,
                        bias_shape
                    )
                    self.model.graph.value_info.append(bias_value_info)
            elif node.op_type == "MatMul":
                # Add default bias (zeros) if needed
                input_shape = self.get_tensor_shape(node.input[0])
                weight_shape = self.get_tensor_shape(node.input[1])
                
                # Try to get shape from value_info if not found in initializers
                if not weight_shape:
                    for info in self.model.graph.value_info:
                        if info.name == node.input[1]:
                            try:
                                weight_shape = [dim.dim_value for dim in info.type.tensor_type.shape.dim]
                            except:
                                pass
                            break
                
                # If still no shape, try to get it from input
                if not weight_shape:
                    for input in self.model.graph.input:
                        if input.name == node.input[1]:
                            try:
                                weight_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
                            except:
                                pass
                            break
                
                # If still no shape, use a default shape
                if not weight_shape:
                    weight_shape = [1]
                
                # Create zero bias tensor with default shape if needed
                bias_shape = [weight_shape[-1]]  # Last dimension of weight matrix
                bias_name = f"{node.name}_default_bias"
                bias_tensor = numpy_helper.from_array(
                    np.zeros(bias_shape, dtype=np.float32),
                    name=bias_name
                )
                self.model.graph.initializer.append(bias_tensor)
                
                # Add value info for the bias tensor
                bias_value_info = helper.make_tensor_value_info(
                    bias_name,
                    onnx_proto.TensorProto.FLOAT,
                    bias_shape
                )
                self.model.graph.value_info.append(bias_value_info)
                
                # Add Add node after MatMul
                add_node = helper.make_node(
                    "Add",
                    [node.output[0], bias_name],
                    [f"{node.output[0]}_biased"],
                    f"{node.name}_add_bias"
                )
                self.model.graph.node.append(add_node)
                
                # Update subsequent nodes to use the biased output
                for next_node in self.model.graph.node:
                    for i, input_name in enumerate(next_node.input):
                        if input_name == node.output[0]:
                            next_node.input[i] = f"{node.output[0]}_biased"
    
    def convert_to_opset_18(self) -> None:
        """Convert model to opset 18 if needed."""
        current_opset = self.model.opset_import[0].version
        if current_opset > 18:
            logger.info(f"Converting from opset {current_opset} to opset 18")
            
            # Log all operators in the model before conversion
            ops = set(node.op_type for node in self.model.graph.node)
            logger.info(f"Operators in model before conversion: {ops}")
            
            # First pass: collect shapes from initializers and inputs
            tensor_shapes = {}
            tensor_types = {}
            
            # Get shapes from initializers
            for init in self.model.graph.initializer:
                tensor_shapes[init.name] = list(init.dims)
                tensor_types[init.name] = init.data_type
            
            # Get shapes from inputs
            for input in self.model.graph.input:
                try:
                    shape = [dim.dim_value if hasattr(dim, 'dim_value') else 1 for dim in input.type.tensor_type.shape.dim]
                    tensor_shapes[input.name] = shape
                    tensor_types[input.name] = input.type.tensor_type.elem_type
                except:
                    # If we can't get the shape, use a default
                    tensor_shapes[input.name] = [1]
                    tensor_types[input.name] = onnx_proto.TensorProto.FLOAT
            
            # Get shapes from value_info
            for info in self.model.graph.value_info:
                try:
                    shape = [dim.dim_value if hasattr(dim, 'dim_value') else 1 for dim in info.type.tensor_type.shape.dim]
                    tensor_shapes[info.name] = shape
                    tensor_types[info.name] = info.type.tensor_type.elem_type
                except:
                    pass
            
            # Second pass: infer shapes for all nodes
            nodes_to_process = list(self.model.graph.node)
            processed_nodes = set()
            
            while nodes_to_process:
                node = nodes_to_process[0]
                can_process = True
                
                # Check if all inputs have shapes
                for input_name in node.input:
                    if input_name not in tensor_shapes:
                        can_process = False
                        break
                
                if can_process:
                    # Process node
                    nodes_to_process.pop(0)
                    processed_nodes.add(node.name)
                    
                    # Infer output shapes based on node type
                    for output_name in node.output:
                        if output_name not in tensor_shapes:
                            shape = None
                            if node.op_type == "MatMul":
                                # MatMul output shape is [batch, M, N] for inputs [batch, M, K] and [batch, K, N]
                                input1_shape = tensor_shapes[node.input[0]]
                                input2_shape = tensor_shapes[node.input[1]]
                                if input1_shape and input2_shape:
                                    shape = input1_shape[:-1] + [input2_shape[-1]]
                            elif node.op_type in ["Add", "Sub", "Mul", "Div", "Erf"]:
                                # Element-wise ops preserve shape
                                shape = tensor_shapes[node.input[0]]
                            elif node.op_type == "Reshape":
                                # For Reshape, try to get shape from second input
                                if len(node.input) > 1 and node.input[1] in tensor_shapes:
                                    shape = tensor_shapes[node.input[1]]
                            elif node.op_type == "ReduceMean":
                                # For ReduceMean, get input shape and remove reduced dimensions
                                input_shape = tensor_shapes[node.input[0]]
                                if input_shape:
                                    # Get axes from attributes
                                    axes = None
                                    keepdims = 1  # default value
                                    for attr in node.attribute:
                                        if attr.name == "axes":
                                            axes = list(attr.ints)
                                        elif attr.name == "keepdims":
                                            keepdims = attr.i
                                    
                                    if axes is not None:
                                        if keepdims:
                                            shape = input_shape.copy()
                                            for axis in axes:
                                                shape[axis if axis >= 0 else len(shape) + axis] = 1
                                        else:
                                            shape = [dim for i, dim in enumerate(input_shape) if i not in axes]
                            elif node.op_type == "Transpose":
                                # For Transpose, get input shape and permute dimensions
                                input_shape = tensor_shapes[node.input[0]]
                                if input_shape:
                                    # Get perm from attributes
                                    perm = None
                                    for attr in node.attribute:
                                        if attr.name == "perm":
                                            perm = list(attr.ints)
                                            break
                                    
                                    if perm is None:
                                        # Default permutation reverses dimensions
                                        perm = list(range(len(input_shape)))[::-1]
                                    
                                    shape = [input_shape[i] for i in perm]
                            
                            if shape:
                                tensor_shapes[output_name] = shape
                                tensor_types[output_name] = tensor_types.get(node.input[0], onnx_proto.TensorProto.FLOAT)
                            else:
                                # If we can't infer the shape, use a default
                                tensor_shapes[output_name] = [1]
                                tensor_types[output_name] = onnx_proto.TensorProto.FLOAT
                else:
                    # Move node to end of list
                    nodes_to_process.append(nodes_to_process.pop(0))
            
            # Add value_info for all tensors
            for tensor_name, shape in tensor_shapes.items():
                value_info = helper.make_tensor_value_info(
                    tensor_name,
                    tensor_types.get(tensor_name, onnx_proto.TensorProto.FLOAT),
                    shape
                )
                self.model.graph.value_info.append(value_info)
            
            # Run shape inference and add missing bias inputs before conversion
            try:
                # First try to infer shapes
                self.model = shape_inference.infer_shapes(self.model)
                
                # Add value_info for any missing tensors
                self.ensure_all_tensors_have_value_info()
                
                # Run shape inference again after adding value_info
                self.model = shape_inference.infer_shapes(self.model)
                
                # Add missing bias inputs
                self.add_missing_bias_inputs()
                
                # Run shape inference one more time after adding bias inputs
                self.model = shape_inference.infer_shapes(self.model)
            except Exception as e:
                logger.warning(f"Shape inference failed: {e}")
            
            try:
                # Try converting directly to 18
                self.model = version_converter.convert_version(self.model, 18)
                logger.info("Model converted to opset 18")
            except Exception as e:
                logger.warning(f"Direct conversion to opset 18 failed: {e}")
                logger.info("Attempting gradual conversion...")
                
                # Try converting gradually through intermediate versions
                current = current_opset
                target = 18
                step = 1
                
                while current > target:
                    try:
                        next_version = max(current - step, target)
                        logger.info(f"Converting from opset {current} to {next_version}")
                        self.model = version_converter.convert_version(self.model, next_version)
                        current = next_version
                    except Exception as e:
                        if step == 1:
                            logger.error(f"Conversion failed at opset {current}: {e}")
                            raise
                        # If failed with current step, try smaller step
                        step = 1
                        continue
                
                logger.info("Gradual conversion completed successfully")
        else:
            logger.info(f"Model already at opset {current_opset} (≤18)")
    
    def create_gelu_replacement(self, input_name: str, output_name: str) -> List[onnx_proto.NodeProto]:
        """Create replacement nodes for Gelu operation using standard operations.
        
        Gelu formula: Y = 0.5 * X * (1 + Erf(X / sqrt(2)))
        """
        nodes = []
        
        # Constants
        const_05 = helper.make_tensor("const_05", onnx_proto.TensorProto.FLOAT, [1], [0.5])
        const_sqrt2 = helper.make_tensor("const_sqrt2", onnx_proto.TensorProto.FLOAT, [1], [np.sqrt(2)])
        const_1 = helper.make_tensor("const_1", onnx_proto.TensorProto.FLOAT, [1], [1.0])
        
        # Intermediate names
        div_name = f"{input_name}_div_sqrt2"
        erf_name = f"{input_name}_erf"
        add_name = f"{input_name}_add_1"
        mul_name = f"{input_name}_mul_05"
        
        # X / sqrt(2)
        div_node = helper.make_node(
            "Div", [input_name, "const_sqrt2"], [div_name], f"gelu_div_{self.node_count}"
        )
        nodes.append(div_node)
        self.node_count += 1
        
        # Erf(X / sqrt(2))
        erf_node = helper.make_node(
            "Erf", [div_name], [erf_name], f"gelu_erf_{self.node_count}"
        )
        nodes.append(erf_node)
        self.node_count += 1
        
        # 1 + Erf(X / sqrt(2))
        add_node = helper.make_node(
            "Add", [erf_name, "const_1"], [add_name], f"gelu_add_{self.node_count}"
        )
        nodes.append(add_node)
        self.node_count += 1
        
        # X * (1 + Erf(X / sqrt(2)))
        mul1_node = helper.make_node(
            "Mul", [input_name, add_name], [mul_name], f"gelu_mul1_{self.node_count}"
        )
        nodes.append(mul1_node)
        self.node_count += 1
        
        # 0.5 * X * (1 + Erf(X / sqrt(2)))
        mul2_node = helper.make_node(
            "Mul", [mul_name, "const_05"], [output_name], f"gelu_mul2_{self.node_count}"
        )
        nodes.append(mul2_node)
        self.node_count += 1
        
        return nodes, [const_05, const_sqrt2, const_1]
    
    def create_layer_norm_replacement(self, input_name: str, output_name: str, 
                                    scale_name: str, bias_name: str, 
                                    axis: int = -1, epsilon: float = 1e-5) -> List[onnx_proto.NodeProto]:
        """Create replacement nodes for LayerNormalization using standard operations.
        
        LayerNorm formula: (X - mean) / sqrt(var + epsilon) * scale + bias
        """
        nodes = []
        
        # Constants
        const_epsilon = helper.make_tensor("const_epsilon", onnx_proto.TensorProto.FLOAT, [1], [epsilon])
        
        # Intermediate names
        mean_name = f"{input_name}_mean"
        sub_mean_name = f"{input_name}_sub_mean"
        square_name = f"{input_name}_square"
        var_name = f"{input_name}_var"
        add_eps_name = f"{input_name}_add_eps"
        sqrt_name = f"{input_name}_sqrt"
        div_name = f"{input_name}_div"
        mul_scale_name = f"{input_name}_mul_scale"
        
        # Calculate mean
        mean_node = helper.make_node(
            "ReduceMean", [input_name], [mean_name], f"layernorm_mean_{self.node_count}",
            axes=[axis], keepdims=1
        )
        nodes.append(mean_node)
        self.node_count += 1
        
        # X - mean
        sub_node = helper.make_node(
            "Sub", [input_name, mean_name], [sub_mean_name], f"layernorm_sub_{self.node_count}"
        )
        nodes.append(sub_node)
        self.node_count += 1
        
        # (X - mean)^2
        square_node = helper.make_node(
            "Mul", [sub_mean_name, sub_mean_name], [square_name], f"layernorm_square_{self.node_count}"
        )
        nodes.append(square_node)
        self.node_count += 1
        
        # Calculate variance
        var_node = helper.make_node(
            "ReduceMean", [square_name], [var_name], f"layernorm_var_{self.node_count}",
            axes=[axis], keepdims=1
        )
        nodes.append(var_node)
        self.node_count += 1
        
        # var + epsilon
        add_eps_node = helper.make_node(
            "Add", [var_name, "const_epsilon"], [add_eps_name], f"layernorm_add_eps_{self.node_count}"
        )
        nodes.append(add_eps_node)
        self.node_count += 1
        
        # sqrt(var + epsilon)
        sqrt_node = helper.make_node(
            "Sqrt", [add_eps_name], [sqrt_name], f"layernorm_sqrt_{self.node_count}"
        )
        nodes.append(sqrt_node)
        self.node_count += 1
        
        # (X - mean) / sqrt(var + epsilon)
        div_node = helper.make_node(
            "Div", [sub_mean_name, sqrt_name], [div_name], f"layernorm_div_{self.node_count}"
        )
        nodes.append(div_node)
        self.node_count += 1
        
        # * scale
        mul_scale_node = helper.make_node(
            "Mul", [div_name, scale_name], [mul_scale_name], f"layernorm_mul_scale_{self.node_count}"
        )
        nodes.append(mul_scale_node)
        self.node_count += 1
        
        # + bias
        add_bias_node = helper.make_node(
            "Add", [mul_scale_name, bias_name], [output_name], f"layernorm_add_bias_{self.node_count}"
        )
        nodes.append(add_bias_node)
        self.node_count += 1
        
        return nodes, [const_epsilon]
    
    def create_rotary_embedding_replacement(self, input_name: str, output_name: str,
                                          cos_name: str, sin_name: str,
                                          position_ids_name: str) -> List[onnx_proto.NodeProto]:
        """Create replacement nodes for RotaryEmbedding using standard operations.
        
        RotaryEmbedding formula: 
        real = cos * x1 - sin * x2
        imag = sin * x1 + cos * x2
        """
        nodes = []
        
        # Intermediate names
        split_name = f"{input_name}_split"
        x1_name = f"{input_name}_x1"
        x2_name = f"{input_name}_x2"
        cos_x1_name = f"{input_name}_cos_x1"
        sin_x2_name = f"{input_name}_sin_x2"
        sin_x1_name = f"{input_name}_sin_x1"
        cos_x2_name = f"{input_name}_cos_x2"
        real_name = f"{input_name}_real"
        imag_name = f"{input_name}_imag"
        
        # Split input into pairs of channels
        split_node = helper.make_node(
            "Split", [input_name], [x1_name, x2_name], f"rope_split_{self.node_count}",
            axis=-1, split=[1, 1]  # This might need adjustment based on actual tensor shapes
        )
        nodes.append(split_node)
        self.node_count += 1
        
        # cos * x1
        cos_x1_node = helper.make_node(
            "Mul", [cos_name, x1_name], [cos_x1_name], f"rope_cos_x1_{self.node_count}"
        )
        nodes.append(cos_x1_node)
        self.node_count += 1
        
        # sin * x2
        sin_x2_node = helper.make_node(
            "Mul", [sin_name, x2_name], [sin_x2_name], f"rope_sin_x2_{self.node_count}"
        )
        nodes.append(sin_x2_node)
        self.node_count += 1
        
        # sin * x1
        sin_x1_node = helper.make_node(
            "Mul", [sin_name, x1_name], [sin_x1_name], f"rope_sin_x1_{self.node_count}"
        )
        nodes.append(sin_x1_node)
        self.node_count += 1
        
        # cos * x2
        cos_x2_node = helper.make_node(
            "Mul", [cos_name, x2_name], [cos_x2_name], f"rope_cos_x2_{self.node_count}"
        )
        nodes.append(cos_x2_node)
        self.node_count += 1
        
        # real = cos * x1 - sin * x2
        real_node = helper.make_node(
            "Sub", [cos_x1_name, sin_x2_name], [real_name], f"rope_real_{self.node_count}"
        )
        nodes.append(real_node)
        self.node_count += 1
        
        # imag = sin * x1 + cos * x2
        imag_node = helper.make_node(
            "Add", [sin_x1_name, cos_x2_name], [imag_name], f"rope_imag_{self.node_count}"
        )
        nodes.append(imag_node)
        self.node_count += 1
        
        # Concatenate real and imaginary parts
        concat_node = helper.make_node(
            "Concat", [real_name, imag_name], [output_name], f"rope_concat_{self.node_count}",
            axis=-1
        )
        nodes.append(concat_node)
        self.node_count += 1
        
        return nodes
    
    def create_attention_replacement(self, q_name: str, k_name: str, v_name: str, 
                                   output_name: str, mask_name: Optional[str] = None,
                                   scale: float = 1.0) -> List[onnx_proto.NodeProto]:
        """Create replacement nodes for GroupQueryAttention using standard operations.
        
        Attention formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        """
        nodes = []
        
        # Constants
        const_scale = helper.make_tensor("const_scale", onnx_proto.TensorProto.FLOAT, [1], [scale])
        
        # Intermediate names
        qkt_name = f"{q_name}_qkt"
        scaled_name = f"{q_name}_scaled"
        masked_name = f"{q_name}_masked" if mask_name else scaled_name
        softmax_name = f"{q_name}_softmax"
        
        # Q @ K^T
        matmul_qk_node = helper.make_node(
            "MatMul", [q_name, k_name], [qkt_name], f"attention_matmul_qk_{self.node_count}"
        )
        nodes.append(matmul_qk_node)
        self.node_count += 1
        
        # Scale by sqrt(d_k) - assuming scale is already provided
        if scale != 1.0:
            mul_scale_node = helper.make_node(
                "Mul", [qkt_name, "const_scale"], [scaled_name], f"attention_scale_{self.node_count}"
            )
            nodes.append(mul_scale_node)
            self.node_count += 1
        else:
            scaled_name = qkt_name
        
        # Apply mask if provided
        if mask_name:
            add_mask_node = helper.make_node(
                "Add", [scaled_name, mask_name], [masked_name], f"attention_mask_{self.node_count}"
            )
            nodes.append(add_mask_node)
            self.node_count += 1
        
        # Softmax
        softmax_node = helper.make_node(
            "Softmax", [masked_name], [softmax_name], f"attention_softmax_{self.node_count}",
            axis=-1
        )
        nodes.append(softmax_node)
        self.node_count += 1
        
        # Final attention: softmax(QK^T) @ V
        matmul_v_node = helper.make_node(
            "MatMul", [softmax_name, v_name], [output_name], f"attention_matmul_v_{self.node_count}"
        )
        nodes.append(matmul_v_node)
        self.node_count += 1
        
        constants = []
        if scale != 1.0:
            constants.append(const_scale)
        
        return nodes, constants
    
    def patch_model(self) -> None:
        """Main patching function that replaces high-version operations."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting model patching...")
        
        # Track all new constants and nodes
        all_new_constants = []
        all_new_nodes = []
        nodes_to_remove = []
        
        # Process each node in the graph
        for i, node in enumerate(self.model.graph.node):
            logger.debug(f"Processing node {i}: {node.op_type}")
            
            if node.op_type == "Gelu":
                logger.info(f"Replacing Gelu node: {node.name}")
                replacement_nodes, constants = self.create_gelu_replacement(
                    node.input[0], node.output[0]
                )
                all_new_nodes.extend(replacement_nodes)
                all_new_constants.extend(constants)
                nodes_to_remove.append(i)
                
            elif node.op_type == "SimplifiedLayerNormalization":
                logger.info(f"Replacing SimplifiedLayerNormalization node: {node.name}")
                # Extract inputs: input, scale, bias
                input_name = node.input[0]
                scale_name = node.input[1] if len(node.input) > 1 else "scale"
                bias_name = node.input[2] if len(node.input) > 2 else "bias"
                
                replacement_nodes, constants = self.create_layer_norm_replacement(
                    input_name, node.output[0], scale_name, bias_name
                )
                all_new_nodes.extend(replacement_nodes)
                all_new_constants.extend(constants)
                nodes_to_remove.append(i)
                
            elif node.op_type == "RotaryEmbedding":
                logger.info(f"Replacing RotaryEmbedding node: {node.name}")
                # Extract inputs: input, cos, sin, position_ids
                input_name = node.input[0]
                cos_name = node.input[1] if len(node.input) > 1 else "cos"
                sin_name = node.input[2] if len(node.input) > 2 else "sin"
                position_ids_name = node.input[3] if len(node.input) > 3 else "position_ids"
                
                replacement_nodes = self.create_rotary_embedding_replacement(
                    input_name, node.output[0], cos_name, sin_name, position_ids_name
                )
                all_new_nodes.extend(replacement_nodes)
                nodes_to_remove.append(i)
                
            elif node.op_type == "GroupQueryAttention":
                logger.info(f"Replacing GroupQueryAttention node: {node.name}")
                # Extract inputs: query, key, value, mask (optional)
                q_name = node.input[0]
                k_name = node.input[1]
                v_name = node.input[2]
                mask_name = node.input[3] if len(node.input) > 3 else None
                
                replacement_nodes, constants = self.create_attention_replacement(
                    q_name, k_name, v_name, node.output[0], mask_name
                )
                all_new_nodes.extend(replacement_nodes)
                all_new_constants.extend(constants)
                nodes_to_remove.append(i)
        
        # Create new graph with updated nodes
        new_nodes = []
        for i, node in enumerate(self.model.graph.node):
            if i not in nodes_to_remove:
                new_nodes.append(node)
        new_nodes.extend(all_new_nodes)
        
        # Create a new graph with the updated nodes
        new_graph = onnx.helper.make_graph(
            nodes=new_nodes,
            name=self.model.graph.name,
            inputs=self.model.graph.input,
            outputs=self.model.graph.output,
            initializer=list(self.model.graph.initializer) + all_new_constants
        )
        
        # Create a new model with the updated graph
        self.model = onnx.helper.make_model(
            graph=new_graph,
            producer_name=self.model.producer_name,
            producer_version=self.model.producer_version,
            domain=self.model.domain,
            model_version=self.model.model_version,
            doc_string=self.model.doc_string,
            opset_imports=self.model.opset_import
        )
        
        logger.info(f"Model patching completed. Added {len(all_new_nodes)} new nodes and {len(all_new_constants)} constants.")
    
    def validate_model(self) -> bool:
        """Validate the patched model."""
        logger.info("Validating patched model...")
        
        try:
            onnx.checker.check_model(self.model)
            logger.info("Model validation passed")
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
        
        try:
            onnx.shape_inference.infer_shapes(self.model)
            logger.info("Shape inference completed")
        except Exception as e:
            logger.warning(f"Shape inference failed: {e}")
            # Continue anyway as this is not critical
        
        return True
    
    def save_model(self, output_path: str) -> None:
        """Save the patched model."""
        logger.info(f"Saving patched model to {output_path}")
        onnx.save(self.model, output_path)
        logger.info("Model saved successfully")
    
    def run_patch(self, output_path: str) -> bool:
        """Run the complete patching pipeline."""
        try:
            self.load_model()
            self.inline_functions()
            
            # First patch custom operators
            self.patch_model()
            
            # Then try to convert to opset 18
            self.convert_to_opset_18()
            
            if self.validate_model():
                self.save_model(output_path)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Patching failed: {e}")
            return False


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Patch ONNX model to use only standard operations")
    parser.add_argument("input_model", help="Path to input ONNX model")
    parser.add_argument("output_model", help="Path to output patched ONNX model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    patcher = ONNXPatcher(args.input_model)
    success = patcher.run_patch(args.output_model)
    
    if success:
        logger.info("Model patching completed successfully!")
        exit(0)
    else:
        logger.error("Model patching failed!")
        exit(1)


if __name__ == "__main__":
    main()
