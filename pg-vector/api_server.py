"""REST API server for pg-vector service."""

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

import service

app = FastAPI(title="pg-vector API", version="1.0.0")

_EMBEDDING_DIM = 512


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


# ── Pydantic models ──────────────────────────────────────────────────


class ProjectCreate(BaseModel):
    researcher_name: str
    topic: str


class ProjectUpdate(BaseModel):
    researcher_name: str | None = None
    topic: str | None = None


class ImageCreate(BaseModel):
    source_type: str = "local"
    lat: float | None = None
    lng: float | None = None
    local_path: str | None = None
    remote_url: str | None = None
    checksum: str | None = None
    embedding: list[float] | None = None

    @field_validator("embedding")
    @classmethod
    def check_embedding_dim(cls, v):
        if v is not None and len(v) != _EMBEDDING_DIM:
            raise ValueError(f"Embedding must be {_EMBEDDING_DIM}-dimensional, got {len(v)}")
        return v


class EmbeddingUpdate(BaseModel):
    embedding: list[float]

    @field_validator("embedding")
    @classmethod
    def check_embedding_dim(cls, v):
        if len(v) != _EMBEDDING_DIM:
            raise ValueError(f"Embedding must be {_EMBEDDING_DIM}-dimensional, got {len(v)}")
        return v


class ResultCreate(BaseModel):
    project_id: int
    image_id: int
    tool_name: str
    result_json: dict | None = None
    confidence: float | None = None
    model_version: str | None = None


class SimilarSearch(BaseModel):
    embedding: list[float]
    top_k: int = 10
    threshold: float | None = None

    @field_validator("embedding")
    @classmethod
    def check_embedding_dim(cls, v):
        if len(v) != _EMBEDDING_DIM:
            raise ValueError(f"Embedding must be {_EMBEDDING_DIM}-dimensional, got {len(v)}")
        return v


# ── Projects ─────────────────────────────────────────────────────────


@app.post("/projects", status_code=201)
def create_project(body: ProjectCreate):
    return JSONResponse(status_code=201, content=service.create_project(body.researcher_name, body.topic))


@app.get("/projects")
def list_projects():
    return JSONResponse(content=service.list_projects())


@app.get("/projects/{project_id}")
def get_project(project_id: int):
    return JSONResponse(content=service.get_project(project_id))


@app.put("/projects/{project_id}")
def update_project(project_id: int, body: ProjectUpdate):
    return JSONResponse(content=service.update_project(project_id, body.researcher_name, body.topic))


@app.delete("/projects/{project_id}")
def delete_project(project_id: int):
    return JSONResponse(content=service.delete_project(project_id))


@app.get("/projects/{project_id}/summary")
def get_project_summary(project_id: int):
    return JSONResponse(content=service.get_project_summary(project_id))


@app.get("/projects/{project_id}/images")
def get_project_images(project_id: int):
    return JSONResponse(content=service.get_project_images(project_id))


# ── Images ───────────────────────────────────────────────────────────


@app.post("/images", status_code=201)
def create_image(body: ImageCreate):
    return JSONResponse(
        status_code=201,
        content=service.create_image(
            body.source_type, body.lat, body.lng,
            body.local_path, body.remote_url, body.checksum, body.embedding,
        ),
    )


@app.get("/images")
def list_images(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    return JSONResponse(content=service.list_images(limit, offset))


@app.get("/images/{image_id}")
def get_image(image_id: int):
    return JSONResponse(content=service.get_image(image_id))


@app.put("/images/{image_id}/embedding")
def update_image_embedding(image_id: int, body: EmbeddingUpdate):
    return JSONResponse(content=service.update_image_embedding(image_id, body.embedding))


@app.delete("/images/{image_id}")
def delete_image(image_id: int):
    return JSONResponse(content=service.delete_image(image_id))


# ── Analysis Results ─────────────────────────────────────────────────


@app.post("/results", status_code=201)
def create_analysis_result(body: ResultCreate):
    return JSONResponse(
        status_code=201,
        content=service.create_analysis_result(
            body.project_id, body.image_id, body.tool_name,
            body.result_json, body.confidence, body.model_version,
        ),
    )


@app.get("/results")
def list_analysis_results(
    project_id: int | None = Query(None),
    image_id: int | None = Query(None),
    tool_name: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    return JSONResponse(content=service.list_analysis_results(project_id, image_id, tool_name, limit, offset))


@app.get("/results/{result_id}")
def get_analysis_result(result_id: int):
    return JSONResponse(content=service.get_analysis_result(result_id))


@app.delete("/results/{result_id}")
def delete_analysis_result(result_id: int):
    return JSONResponse(content=service.delete_analysis_result(result_id))


# ── Vector Search ────────────────────────────────────────────────────


@app.post("/search/similar")
def search_similar(body: SimilarSearch):
    return JSONResponse(content=service.search_similar_images(body.embedding, body.top_k, body.threshold))


@app.get("/search/similar/{image_id}")
def search_similar_by_image(
    image_id: int,
    top_k: int = Query(10, ge=1, le=100),
    threshold: float | None = Query(None),
):
    return JSONResponse(content=service.search_similar_by_image_id(image_id, top_k, threshold))
