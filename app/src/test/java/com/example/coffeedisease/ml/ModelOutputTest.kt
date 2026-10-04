package com.example.coffeedisease.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class ModelOutputTest {
    @Test fun preservesMultipleLabelsAndThresholdBoundary() {
        assertEquals("disease_cercospora,disease_phoma", ModelOutput.decode(floatArrayOf(0.5f, 0.1f, 0.9f, 0.49f)))
    }
    @Test fun noDetectionIsNotHealthy() {
        assertEquals("disease_none", ModelOutput.decode(floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)))
    }
    @Test fun preservesExportLabelOrder() {
        assertEquals("disease_rust", ModelOutput.decode(floatArrayOf(0f, 0f, 0f, 1f)))
    }
    @Test(expected = IllegalArgumentException::class) fun rejectsInvalidScores() {
        ModelOutput.decode(floatArrayOf(Float.NaN, 0f, 0f, 0f))
    }
}
