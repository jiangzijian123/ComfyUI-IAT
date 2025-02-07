import node_helpers

class CLIPTextEncodeFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning/flux"

    def encode(self, clip, clip_l, t5xxl, guidance):
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        return (clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance}), )

class CLIPTextEncodeFluxIAT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            # "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            # "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip_l_input": ("STRING", {"forceInput": True}),
            "t5xxl_input": ("STRING", {"forceInput": True} ),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
        }}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning/flux"

    def encode(self, clip, clip_l_input, t5xxl_input, guidance):
        # tokens = clip.tokenize(clip_l_input)
        # tokens["t5xxl"] = clip.tokenize(t5xxl_input)["t5xxl"]
        print(type(clip_l_input))
        print('clip_l_input:', clip_l_input)
        print(t5xxl_input)
        tokens = clip.tokenize(clip_l_input)
        tokens["t5xxl"] = clip.tokenize(t5xxl_input)["t5xxl"]

        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        output["guidance"] = guidance
        return ([[cond, output]], )

class FluxGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING", ),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "advanced/conditioning/flux"

    def append(self, conditioning, guidance):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (c, )


class FluxDisableGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING", ),
            }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "advanced/conditioning/flux"
    DESCRIPTION = "This node completely disables the guidance embed on Flux and Flux like models"

    def append(self, conditioning):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": None})
        return (c, )


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeFlux": CLIPTextEncodeFlux,
    "FluxGuidance": FluxGuidance,
    "FluxDisableGuidance": FluxDisableGuidance,
    "CLIPTextEncodeFluxIAT":CLIPTextEncodeFluxIAT
}
