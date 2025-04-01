
-- https://www.geeksforgeeks.org/making-a-flask-app-using-a-postgresql-database/

CREATE TABLE users (
    id SERIAL PRIMARY KEY,                           -- Auto-incrementing user ID
    username VARCHAR(50) NOT NULL UNIQUE,             -- Unique username, cannot be null
    password VARCHAR(255) NOT NULL,                   -- Password, cannot be null
    email VARCHAR(255) UNIQUE,                        -- Unique email
    role VARCHAR(50) DEFAULT 'user',                  -- User role, default is 'user'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- Timestamp for when the record was created
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- Timestamp for when the record was last updated
    first_name VARCHAR(50) NOT NULL,                  -- First name of the user
    last_name VARCHAR(50) NOT NULL                    -- Last name of the user
);



CREATE TABLE blog_posts (
  id SERIAL PRIMARY KEY,
  title VARCHAR(255),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
