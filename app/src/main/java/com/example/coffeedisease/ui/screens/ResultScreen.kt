package com.example.coffeedisease.ui.screens

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.os.Build
import android.provider.MediaStore
import android.net.Uri
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.coffeedisease.R
import com.example.coffeedisease.data.AppDatabase
import com.example.coffeedisease.data.ScanResult
import com.example.coffeedisease.ml.DiseaseClassifier
import com.example.coffeedisease.ml.ScreeningOutcome
import com.example.coffeedisease.ui.theme.StatusDisease
import com.example.coffeedisease.ui.theme.StatusInfo
import com.example.coffeedisease.ui.theme.StatusWarning
import kotlinx.coroutines.CancellationException

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(profileId: Int, imageUri: String, onDone: () -> Unit) {
    val context = LocalContext.current
    val database = AppDatabase.getDatabase(context)
    val classifier = remember { DiseaseClassifier(context) }
    var outcome by remember { mutableStateOf<ScreeningOutcome?>(null) }
    var analysisFailed by remember { mutableStateOf(false) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }

    LaunchedEffect(imageUri) {
        try {
            val uri = Uri.parse(imageUri)
            bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { decoder, _, _ ->
                    decoder.isMutableRequired = true
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
            }
            val captured = checkNotNull(bitmap) { "Could not decode captured photo" }
            val result = classifier.classifyImage(captured)
            database.scanResultDao().insert(
                ScanResult(profileId = profileId, diseaseResult = result.historyKey, imageUri = imageUri)
            )
            outcome = result
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            android.util.Log.e("DiseaseClassifier", "Photo screening failed", error)
            analysisFailed = true
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.result_title)) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    titleContentColor = MaterialTheme.colorScheme.onPrimary
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier.fillMaxSize().padding(padding).padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            bitmap?.let { image ->
                Image(
                    bitmap = image.asImageBitmap(),
                    contentDescription = stringResource(R.string.captured_coffee_leaf),
                    modifier = Modifier.fillMaxWidth().aspectRatio(1f).clip(RoundedCornerShape(16.dp))
                        .background(MaterialTheme.colorScheme.surfaceVariant)
                )
            }
            Spacer(Modifier.height(20.dp))

            when (val result = outcome) {
                null -> if (analysisFailed) {
                    Text(stringResource(R.string.analysis_error), style = MaterialTheme.typography.titleLarge)
                } else {
                    CircularProgressIndicator()
                    Text(stringResource(R.string.analyzing_photo), modifier = Modifier.padding(top = 16.dp))
                }
                ScreeningOutcome.NotCoffeeLeaf -> OutcomeCard(
                    title = stringResource(R.string.not_coffee_leaf),
                    detail = null,
                    color = StatusInfo
                )
                ScreeningOutcome.RetakePhoto -> OutcomeCard(
                    title = stringResource(R.string.retake_photo),
                    detail = null,
                    color = StatusWarning
                )
                ScreeningOutcome.NoHealthIssue -> OutcomeCard(
                    title = stringResource(R.string.no_health_issue_detected),
                    detail = stringResource(R.string.screening_caveat),
                    color = StatusInfo
                )
                is ScreeningOutcome.ConditionsDetected -> OutcomeCard(
                    title = result.labels.joinToString(", ") { key ->
                        val resource = context.resources.getIdentifier(key, "string", context.packageName)
                        if (resource != 0) context.getString(resource) else key
                    },
                    detail = stringResource(R.string.screening_detected_caveat),
                    color = StatusDisease
                )
            }

            Spacer(Modifier.weight(1f))
            Button(onClick = onDone, modifier = Modifier.fillMaxWidth().height(64.dp), shape = RoundedCornerShape(16.dp)) {
                Text(stringResource(R.string.done), fontSize = 20.sp, fontWeight = FontWeight.Bold)
            }
        }
    }

    DisposableEffect(classifier) {
        onDispose { classifier.close() }
    }
}

@Composable
private fun OutcomeCard(title: String, detail: String?, color: androidx.compose.ui.graphics.Color) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.12f)),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(Modifier.padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = title,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = color
            )
            detail?.let {
                Spacer(Modifier.height(16.dp))
                Text(it, style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurface)
            }
        }
    }
}
