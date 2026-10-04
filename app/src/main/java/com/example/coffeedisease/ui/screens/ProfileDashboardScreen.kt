package com.example.coffeedisease.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.coffeedisease.R
import com.example.coffeedisease.data.AppDatabase
import com.example.coffeedisease.data.Profile
import com.example.coffeedisease.data.ScanResult
import com.example.coffeedisease.ui.theme.CardBackground
import com.example.coffeedisease.ui.theme.StatusDisease
import com.example.coffeedisease.ui.theme.StatusInfo
import com.example.coffeedisease.ui.theme.StatusWarning
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProfileDashboardScreen(
    profileId: Int,
    onBack: () -> Unit,
    onNewScan: () -> Unit
) {
    val context = LocalContext.current
    val database = AppDatabase.getDatabase(context)

    var profile by remember { mutableStateOf<Profile?>(null) }
    val scans by database.scanResultDao().getScansForProfile(profileId).collectAsState(initial = emptyList())

    LaunchedEffect(profileId) {
        if (profileId != -1) {
            profile = database.profileDao().getProfileById(profileId)
        } else {
            profile = Profile(id = -1, name = context.getString(R.string.shared_profile), emoji = "👥", isShared = true)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(profile?.name ?: "") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    titleContentColor = MaterialTheme.colorScheme.onPrimary
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
        ) {
            Button(
                onClick = onNewScan,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(80.dp),
                shape = RoundedCornerShape(16.dp)
            ) {
                Text(
                    text = "📷 " + stringResource(R.string.take_photo),
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = stringResource(R.string.recent_scans),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(16.dp))

            LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                items(scans) { scan ->
                    ScanResultCard(scan = scan)
                }
            }
        }
    }
}

@Composable
fun ScanResultCard(scan: ScanResult) {
    val isNoIssue = scan.diseaseResult == "disease_none"
    val isNotLeaf = scan.diseaseResult == "not_coffee_leaf"
    val isRetake = scan.diseaseResult == "retake_photo"
    val statusColor = when {
        isNoIssue || isNotLeaf -> StatusInfo
        isRetake -> StatusWarning
        else -> StatusDisease
    }

    Card(
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = statusColor.copy(alpha = 0.1f)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(text = if (isRetake || isNotLeaf) "?" else if (isNoIssue) "i" else "!", fontSize = 32.sp, color = statusColor)
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                val context = LocalContext.current
                val localizedResult = when (scan.diseaseResult) {
                    "not_coffee_leaf" -> context.getString(R.string.history_not_coffee_leaf)
                    "retake_photo" -> context.getString(R.string.history_retake_photo)
                    "disease_none" -> context.getString(R.string.no_health_issue_detected)
                    else -> scan.diseaseResult.split(",").joinToString(", ") { key ->
                        val id = context.resources.getIdentifier(key, "string", context.packageName)
                        if (id != 0) context.getString(id) else key
                    }
                }
                Text(
                    text = localizedResult,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = statusColor
                )
                if (isNoIssue) {
                    Text(
                        text = stringResource(R.string.screening_caveat),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                val sdf = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault())
                Text(
                    text = sdf.format(Date(scan.timestamp)),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}
