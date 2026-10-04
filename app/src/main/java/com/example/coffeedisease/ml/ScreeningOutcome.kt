package com.example.coffeedisease.ml

sealed class ScreeningOutcome(val historyKey: String) {
    data object NotCoffeeLeaf : ScreeningOutcome("not_coffee_leaf")
    data object RetakePhoto : ScreeningOutcome("retake_photo")
    data object NoHealthIssue : ScreeningOutcome("disease_none")
    data class ConditionsDetected(val labels: List<String>) : ScreeningOutcome(labels.joinToString(","))
}

sealed class LeafScreeningDecision {
    data object Rejected : LeafScreeningDecision()
    data object Uncertain : LeafScreeningDecision()
    data object Accepted : LeafScreeningDecision()

    companion object {
        fun fromCoffeeLeafScore(score: Float, rejectThreshold: Float, acceptThreshold: Float): LeafScreeningDecision {
            require(score.isFinite() && score in 0f..1f)
            require(rejectThreshold.isFinite() && acceptThreshold.isFinite())
            require(rejectThreshold in 0f..1f && acceptThreshold in 0f..1f && rejectThreshold < acceptThreshold)
            return when {
                score <= rejectThreshold -> Rejected
                score >= acceptThreshold -> Accepted
                else -> Uncertain
            }
        }
    }
}

object ScreeningFlow {
    fun classify(
        coffeeLeafScore: Float,
        rejectThreshold: Float,
        acceptThreshold: Float,
        diseaseThreshold: Float,
        diseaseLabels: List<String>,
        runDiseaseModel: () -> FloatArray
    ): ScreeningOutcome = when (
        LeafScreeningDecision.fromCoffeeLeafScore(coffeeLeafScore, rejectThreshold, acceptThreshold)
    ) {
        LeafScreeningDecision.Rejected -> ScreeningOutcome.NotCoffeeLeaf
        LeafScreeningDecision.Uncertain -> ScreeningOutcome.RetakePhoto
        LeafScreeningDecision.Accepted -> {
            val detected = ModelOutput.decode(runDiseaseModel(), diseaseThreshold, diseaseLabels)
            if (detected == "disease_none") ScreeningOutcome.NoHealthIssue
            else ScreeningOutcome.ConditionsDetected(detected.split(","))
        }
    }
}
