CREATE DATABASE IF NOT EXISTS security_db CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;

USE security_db;

CREATE TABLE IF NOT EXISTS event (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(20),
    confidence FLOAT,
    motion_score FLOAT,
    image_path VARCHAR(255),
    description TEXT
);
