package com.example.coffeedisease.ml

object ModelOutput {
    private val defaultLabels = listOf("disease_cercospora", "disease_miner", "disease_phoma", "disease_rust")

    fun decode(scores: FloatArray, threshold: Float = 0.5f, labels: List<String> = defaultLabels): String {
        require(scores.size == labels.size && scores.all { it.isFinite() && it in 0f..1f })
        require(threshold.isFinite() && threshold in 0f..1f)
        // Sigmoid is already included in the graph. Preserve every positive label.
        return labels.filterIndexed { index, _ -> scores[index] >= threshold }
            .joinToString(",").ifEmpty { "disease_none" }
    }
}
