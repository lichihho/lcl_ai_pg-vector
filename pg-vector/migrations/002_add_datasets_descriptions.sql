-- Migration 002: Add datasets, dataset_images, and image_descriptions tables
-- Date: 2026-02-15
-- Description: Support dataset curation metadata and LLM text descriptions

-- datasets: registry of image collections
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id    SERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT,
    methodology   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- dataset_images: M:N link between datasets and images with curation metadata
CREATE TABLE IF NOT EXISTS dataset_images (
    dataset_id    INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    image_id      INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    metadata      JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, image_id)
);

CREATE INDEX IF NOT EXISTS idx_dataset_images_image ON dataset_images (image_id);

-- image_descriptions: LLM-generated text descriptions with text embeddings
CREATE TABLE IF NOT EXISTS image_descriptions (
    description_id SERIAL PRIMARY KEY,
    project_id     INTEGER REFERENCES projects(project_id) ON DELETE CASCADE,
    image_id       INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    tool_name      TEXT NOT NULL,
    content        TEXT NOT NULL,
    embedding      vector(1536),
    language       TEXT NOT NULL DEFAULT 'zh',
    model_version  TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_descriptions_image ON image_descriptions (image_id);
CREATE INDEX IF NOT EXISTS idx_descriptions_project ON image_descriptions (project_id);
CREATE INDEX IF NOT EXISTS idx_descriptions_tool ON image_descriptions (tool_name);
CREATE INDEX IF NOT EXISTS idx_descriptions_embedding_hnsw ON image_descriptions
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
