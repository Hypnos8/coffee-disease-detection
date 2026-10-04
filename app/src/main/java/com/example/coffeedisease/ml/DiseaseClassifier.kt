package com.example.coffeedisease.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.roundToInt

class DiseaseClassifier(context: Context) {
    private val assets = context.applicationContext.assets
    private val environment = OrtEnvironment.getEnvironment()
    private var leafSession: OrtSession? = null
    private var conditionSession: OrtSession? = null
    private var closed = false

    private val leafMetadata by lazy {
        JSONObject(assets.open("coffee_leaf_classifier.metadata.json").bufferedReader().use { it.readText() })
    }
    private val conditionMetadata by lazy {
        JSONObject(assets.open("coffee_condition_classifier.metadata.json").bufferedReader().use { it.readText() })
    }
    private val leafLabels by lazy {
        assets.open("coffee_leaf_classifier.labels.txt").bufferedReader().useLines { it.toList() }
    }
    private val conditionLabels by lazy {
        assets.open("coffee_condition_classifier.labels.txt").bufferedReader().useLines { lines ->
            lines.map { "disease_${it.lowercase()}" }.toList()
        }
    }

    suspend fun classifyImage(bitmap: Bitmap): ScreeningOutcome = withContext(Dispatchers.Default) {
        synchronized(this@DiseaseClassifier) {
            check(!closed) { "Classifier is closed" }
            val image = normalize(bitmap)
            val leafScores = runModel(
                modelName = "coffee_leaf_classifier.onnx",
                metadata = leafMetadata,
                input = image,
                getSession = { leafSession },
                setSession = { leafSession = it }
            )
            val thresholds = leafMetadata.getJSONObject("leaf_thresholds")
            ScreeningFlow.classify(
                coffeeLeafScore = leafScores[leafLabels.indexOf("coffee_leaf")],
                rejectThreshold = thresholds.getDouble("reject").toFloat(),
                acceptThreshold = thresholds.getDouble("accept").toFloat(),
                diseaseThreshold = conditionMetadata.getDouble("threshold").toFloat(),
                diseaseLabels = conditionLabels
            ) {
                    runModel(
                        modelName = "coffee_condition_classifier.onnx",
                        metadata = conditionMetadata,
                        input = image,
                        getSession = { conditionSession },
                        setSession = { conditionSession = it }
                    )
            }
        }
    }

    private fun runModel(
        modelName: String,
        metadata: JSONObject,
        input: FloatArray,
        getSession: () -> OrtSession?,
        setSession: (OrtSession) -> Unit
    ): FloatArray {
        val activeSession = getSession() ?: OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(2)
            environment.createSession(assets.open(modelName).use { it.readBytes() }, options)
        }.also(setSession)
        val inputMetadata = metadata.getJSONObject("input")
        val shapeValues = inputMetadata.getJSONArray("shape")
        val shape = LongArray(shapeValues.length()) { shapeValues.getLong(it) }
        require(input.size == shape.fold(1L) { total, dimension -> total * dimension }.toInt())
        val inputName = inputMetadata.getString("name")
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), shape).use { tensor ->
            activeSession.run(mapOf(inputName to tensor)).use { output ->
                @Suppress("UNCHECKED_CAST")
                val values = output[0].value as Array<FloatArray>
                return values[0]
            }
        }
    }

    private fun normalize(source: Bitmap): FloatArray {
        // Both exports use centered, aspect-preserving fit and ImageNet normalization.
        val size = conditionMetadata.getJSONObject("preprocessing").getInt("size")
        val scale = minOf(size.toDouble() / source.width, size.toDouble() / source.height)
        val width = (source.width * scale).roundToInt().coerceIn(1, size)
        val height = (source.height * scale).roundToInt().coerceIn(1, size)
        val resized = Bitmap.createScaledBitmap(source, width, height, true)
        val padded = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(padded).apply {
            drawColor(Color.WHITE)
            drawBitmap(resized, ((size - width) / 2).toFloat(), ((size - height) / 2).toFloat(), null)
        }
        val pixels = IntArray(size * size)
        padded.getPixels(pixels, 0, size, 0, 0, size, size)
        padded.recycle()
        if (resized !== source) resized.recycle()
        val preprocessing = conditionMetadata.getJSONObject("preprocessing")
        val meanJson = preprocessing.getJSONArray("mean")
        val stdJson = preprocessing.getJSONArray("std")
        val mean = FloatArray(3) { meanJson.getDouble(it).toFloat() }
        val std = FloatArray(3) { stdJson.getDouble(it).toFloat() }
        val values = FloatArray(3 * pixels.size)
        for (channel in 0..2) {
            for (index in pixels.indices) {
                val value = (pixels[index] shr (16 - 8 * channel)) and 255
                values[channel * pixels.size + index] = (value / 255f - mean[channel]) / std[channel]
            }
        }
        return values
    }

    @Synchronized
    fun close() {
        closed = true
        leafSession?.close()
        conditionSession?.close()
        leafSession = null
        conditionSession = null
    }
}
