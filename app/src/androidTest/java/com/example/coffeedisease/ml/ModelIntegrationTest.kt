package com.example.coffeedisease.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ModelIntegrationTest {
    @Test fun exportedFixtureMatchesOnDevice() {
        assertModelFixtureMatches(
            modelName = "coffee_condition_classifier.onnx",
            inputName = "coffee_condition_classifier.verification_input.f32",
            expectedName = "coffee_condition_classifier.verification_expected.json",
            outputCount = 4
        ) { scores ->
            assertEquals("disease_miner,disease_phoma", ModelOutput.decode(scores))
        }
    }

    @Test fun coffeeLeafExportFixtureMatchesOnDevice() {
        assertModelFixtureMatches(
            modelName = "coffee_leaf_classifier.onnx",
            inputName = "coffee_leaf_classifier.verification_input.f32",
            expectedName = "coffee_leaf_classifier.verification_expected.json",
            outputCount = 2
        ) { scores ->
            assertEquals(LeafScreeningDecision.Accepted, LeafScreeningDecision.fromCoffeeLeafScore(scores[1], 0.2f, 0.8f))
        }
    }

    private fun assertModelFixtureMatches(
        modelName: String,
        inputName: String,
        expectedName: String,
        outputCount: Int,
        assertSemanticOutput: (FloatArray) -> Unit
    ) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val assets = instrumentation.context.assets
        val model = instrumentation.targetContext.assets.open(modelName).use { it.readBytes() }
        val input = assets.open(inputName).use { it.readBytes() }
        val expected = JSONObject(assets.open(expectedName).bufferedReader().use { it.readText() })
            .getJSONObject("scores").getJSONArray("fp32")
        val environment = OrtEnvironment.getEnvironment()
        OrtSession.SessionOptions().use { options ->
            environment.createSession(model, options).use { session ->
                OnnxTensor.createTensor(environment, ByteBuffer.wrap(input).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer(), longArrayOf(1, 3, 224, 224)).use { tensor ->
                    session.run(mapOf("image" to tensor)).use { result ->
                        @Suppress("UNCHECKED_CAST")
                        val scores = (result[0].value as Array<FloatArray>)[0]
                        assertEquals(outputCount, scores.size)
                        for (index in scores.indices) {
                            assertEquals(expected.getDouble(index), scores[index].toDouble(), 0.0001)
                        }
                        assertSemanticOutput(scores)
                    }
                }
            }
        }
    }
}
