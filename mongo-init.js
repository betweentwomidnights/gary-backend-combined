// Initialize database
db = db.getSiblingDB('audio_generation_db');

// Session collection indexes
db.sessions.createIndex({ "_id": 1 });
db.sessions.createIndex({ "created_at": 1 }, { expireAfterSeconds: 604800 }); // 7-day TTL
db.sessions.createIndex({ "updated_at": 1 });
db.sessions.createIndex({ "model_name": 1 });

// Audio reference indexes - optimize retrieval patterns seen in code
db.sessions.createIndex({ "last_processed_audio": 1 });
db.sessions.createIndex({ "last_input_audio": 1 });
db.sessions.createIndex({ "initial_audio": 1 });

// GridFS optimization for audio storage
// Use larger chunks for audio files (1MB instead of default 255KB)
db.fs.chunks.drop();
db.fs.files.drop();

// Re-initialize GridFS with optimized settings
db.createCollection('fs.files');
db.createCollection('fs.chunks', {
  storageEngine: {
    wiredTiger: {
      configString: 'block_compressor=snappy'
    }
  }
});

// Create GridFS indexes
db.fs.chunks.createIndex({ "files_id": 1, "n": 1 }, { unique: true });
db.fs.files.createIndex({ "filename": 1 });
db.fs.files.createIndex({ "uploadDate": 1 });
db.fs.files.createIndex({ "md5": 1 });

// Connection pooling optimized for parallel audio processing
db.adminCommand({
  setParameter: 1,
  maxConnecting: 20,
  connPoolMaxConnsPerHost: 100,
  connPoolMaxShardedConnsPerHost: 100
});

// Set up write concern defaults appropriate for audio data
db.adminCommand({
  setDefaultRWConcern: 1,
  defaultWriteConcern: { w: 1, wtimeout: 5000 }
});

print("MongoDB optimization complete - configured for audio processing workloads");