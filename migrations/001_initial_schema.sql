-- MoneyTalks initial schema
-- Run this in the Supabase SQL editor once after creating your project.

-- ── Tables ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS "ScannedMoney" (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    denomination VARCHAR(20) NOT NULL,
    confidence  FLOAT       NOT NULL,
    image_path  TEXT,
    scanned_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "Administrator" (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "ModelVersions" (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    version_string VARCHAR(50) NOT NULL,
    file_path      TEXT        NOT NULL,
    uploaded_by    VARCHAR(255),
    uploaded_at    TIMESTAMPTZ DEFAULT NOW(),
    is_deployed    BOOLEAN     DEFAULT FALSE
);

-- ── Row Level Security ───────────────────────────────────────────────────────
-- The service-role key (used server-side) bypasses RLS automatically.
-- These policies block direct public/anon access.

ALTER TABLE "ScannedMoney"  ENABLE ROW LEVEL SECURITY;
ALTER TABLE "Administrator" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "ModelVersions" ENABLE ROW LEVEL SECURITY;

-- Deny all access from the anon / authenticated roles (service role bypasses RLS)
CREATE POLICY "deny_anon_scanned"  ON "ScannedMoney"  FOR ALL TO anon USING (false);
CREATE POLICY "deny_anon_admin"    ON "Administrator"  FOR ALL TO anon USING (false);
CREATE POLICY "deny_anon_models"   ON "ModelVersions"  FOR ALL TO anon USING (false);

-- ── Storage Buckets ──────────────────────────────────────────────────────────
-- Create these manually in the Supabase dashboard → Storage → New bucket:
--   Name: scanned-images   Public: OFF
--   Name: model-files      Public: OFF

-- ── Seed: first admin account ────────────────────────────────────────────────
-- Replace the hash below with the output of:
--   python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt(12)).decode())"
-- INSERT INTO "Administrator" (email, password_hash)
-- VALUES ('admin@moneytalks.app', '$2b$12$REPLACE_WITH_REAL_HASH');
