package com.example.coffeedisease.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "scan_results")
data class ScanResult(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val profileId: Int,
    val timestamp: Long = System.currentTimeMillis(),
    val diseaseResult: String,
    val imageUri: String
)