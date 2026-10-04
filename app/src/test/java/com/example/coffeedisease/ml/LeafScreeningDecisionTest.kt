package com.example.coffeedisease.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LeafScreeningDecisionTest {
    @Test fun rejectsAtAndBelowMetadataRejectThreshold() {
        assertEquals(LeafScreeningDecision.Rejected, LeafScreeningDecision.fromCoffeeLeafScore(0.2f, 0.2f, 0.8f))
        assertEquals(LeafScreeningDecision.Rejected, LeafScreeningDecision.fromCoffeeLeafScore(0.1f, 0.2f, 0.8f))
    }

    @Test fun asksForRetakeBetweenThresholds() {
        assertEquals(LeafScreeningDecision.Uncertain, LeafScreeningDecision.fromCoffeeLeafScore(0.5f, 0.2f, 0.8f))
    }

    @Test fun acceptsAtAndAboveMetadataAcceptThreshold() {
        assertEquals(LeafScreeningDecision.Accepted, LeafScreeningDecision.fromCoffeeLeafScore(0.8f, 0.2f, 0.8f))
        assertEquals(LeafScreeningDecision.Accepted, LeafScreeningDecision.fromCoffeeLeafScore(0.9f, 0.2f, 0.8f))
    }

    @Test fun rejectsInvalidThresholdsAndScores() {
        listOf(floatArrayOf(Float.NaN, 0.2f, 0.8f), floatArrayOf(0.5f, 0.8f, 0.2f)).forEach { values ->
            try {
                LeafScreeningDecision.fromCoffeeLeafScore(values[0], values[1], values[2])
                throw AssertionError("Expected invalid screening inputs to fail")
            } catch (_: IllegalArgumentException) {
                // Expected.
            }
        }
    }

    @Test fun rejectedAndUncertainPhotosSkipDiseaseInference() {
        var conditionModelCalled = false
        val conditionScores = { conditionModelCalled = true; floatArrayOf(0f, 0f, 0f, 0f) }
        assertEquals(
            ScreeningOutcome.NotCoffeeLeaf,
            ScreeningFlow.classify(0.2f, 0.2f, 0.8f, 0.5f, diseaseLabels, conditionScores)
        )
        assertFalse(conditionModelCalled)
        assertEquals(
            ScreeningOutcome.RetakePhoto,
            ScreeningFlow.classify(0.5f, 0.2f, 0.8f, 0.5f, diseaseLabels, conditionScores)
        )
        assertFalse(conditionModelCalled)
    }

    @Test fun acceptedPhotoRunsDiseaseInferenceAndKeepsHealthUnconfirmed() {
        var conditionModelCalled = false
        val outcome = ScreeningFlow.classify(0.8f, 0.2f, 0.8f, 0.5f, diseaseLabels) {
            conditionModelCalled = true
            floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        }
        assertTrue(conditionModelCalled)
        assertEquals(ScreeningOutcome.NoHealthIssue, outcome)
        assertEquals("disease_none", outcome.historyKey)
    }

    private companion object {
        val diseaseLabels = listOf("disease_cercospora", "disease_miner", "disease_phoma", "disease_rust")
    }
}
