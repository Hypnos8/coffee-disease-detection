package com.example.coffeedisease.data

import android.database.sqlite.SQLiteDatabase
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class ScanHistoryMigrationTest {
    @Test fun upgradeClearsOldScanPredictionsAndPreservesProfiles() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = "scan_history_migration_test.db"
        context.deleteDatabase(name)
        val file = context.getDatabasePath(name)
        file.parentFile?.mkdirs()
        SQLiteDatabase.openOrCreateDatabase(file, null).use { oldDb ->
            oldDb.execSQL("CREATE TABLE IF NOT EXISTS `profiles` (`id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, `name` TEXT NOT NULL, `emoji` TEXT NOT NULL, `isShared` INTEGER NOT NULL)")
            oldDb.execSQL("CREATE TABLE IF NOT EXISTS `scan_results` (`id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, `profileId` INTEGER NOT NULL, `timestamp` INTEGER NOT NULL, `diseaseResult` TEXT NOT NULL, `imageUri` TEXT NOT NULL)")
            oldDb.execSQL("CREATE TABLE IF NOT EXISTS room_master_table (id INTEGER PRIMARY KEY,identity_hash TEXT)")
            oldDb.execSQL("INSERT OR REPLACE INTO room_master_table (id,identity_hash) VALUES(42, '5c08716de28ffd0ec1a32aa3021473bf')")
            oldDb.execSQL("INSERT INTO profiles (id, name, emoji, isShared) VALUES (1, 'Amina', '🌱', 0)")
            oldDb.execSQL("INSERT INTO scan_results (profileId, timestamp, diseaseResult, imageUri) VALUES (1, 1, 'disease_rust', 'file:///old.jpg')")
            oldDb.version = 1
        }

        val upgraded = AppDatabase.buildDatabase(context, name)
        try {
            assertEquals(emptyList<ScanResult>(), upgraded.scanResultDao().getScansForProfile(1).first())
            assertNotNull(upgraded.profileDao().getProfileById(1))
        } finally {
            upgraded.close()
            context.deleteDatabase(name)
        }
    }
}
