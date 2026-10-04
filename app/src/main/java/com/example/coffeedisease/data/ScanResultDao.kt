package com.example.coffeedisease.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ScanResultDao {
    @Query("SELECT * FROM scan_results WHERE profileId = :profileId ORDER BY timestamp DESC")
    fun getScansForProfile(profileId: Int): Flow<List<ScanResult>>

    @Insert
    suspend fun insert(scanResult: ScanResult)
}