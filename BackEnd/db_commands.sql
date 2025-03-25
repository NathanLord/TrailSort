
-- https://www.geeksforgeeks.org/making-a-flask-app-using-a-postgresql-database/

CREATE TABLE users (
    id SERIAL PRIMARY KEY,                          
    username VARCHAR(50) NOT NULL UNIQUE,          
    password VARCHAR(255) NOT NULL,                
    email VARCHAR(255) UNIQUE,            
    role VARCHAR(50) DEFAULT 'user',               
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
);


CREATE TABLE blog_posts (
  id SERIAL PRIMARY KEY,
  title VARCHAR(255),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
