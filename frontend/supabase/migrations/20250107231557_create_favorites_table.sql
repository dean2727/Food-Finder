-- Create a table to store user-specific favorite restaurants, including user ID, the name of the favorite, a description of why the user likes it and an optional shared link.
CREATE TABLE "favorites" (
    "id" SERIAL PRIMARY KEY,
    "user_id" INTEGER NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "link" TEXT,
    "created_at" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE
);