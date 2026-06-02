"""
Pydantic schemas for the Gen-Z Translator API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TranslateRequest(BaseModel):
    """Request body for the /translate endpoint."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Normal English text to translate into Gen-Z slang.",
        examples=["I'm really tired today, I need some rest."],
    )
    max_new_tokens: int = Field(
        default=150,
        ge=10,
        le=512,
        description="Maximum number of tokens to generate in the response.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher = more creative, lower = more deterministic.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling. 1.0 = no filtering.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional custom system prompt. If not provided, uses the default Gen-Z translator prompt.",
    )


class TranslateResponse(BaseModel):
    """Response body from the /translate endpoint."""
    input_text: str = Field(description="The original English text submitted.")
    gen_z_text: str = Field(description="The Gen-Z slang translation.")
    model: str = Field(description="The model used for translation.")
    tokens_generated: int = Field(description="Approximate number of tokens in the response.")


class HealthResponse(BaseModel):
    """Response body from the /health endpoint."""
    status: str = Field(description="Service status: 'healthy' or 'loading'.")
    model_loaded: bool = Field(description="Whether the model is loaded in GPU memory.")
    model_path: str = Field(description="Path to the model weights on disk.")
    gpu: Optional[str] = Field(default=None, description="GPU name if CUDA is available.")
    vram_used_mb: Optional[float] = Field(default=None, description="GPU VRAM in use (MB).")
    vram_total_mb: Optional[float] = Field(default=None, description="Total GPU VRAM (MB).")
