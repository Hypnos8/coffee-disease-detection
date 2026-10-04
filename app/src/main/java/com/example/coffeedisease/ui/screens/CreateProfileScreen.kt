package com.example.coffeedisease.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.coffeedisease.R
import com.example.coffeedisease.data.AppDatabase
import com.example.coffeedisease.data.Profile
import kotlinx.coroutines.launch

val Emojis = listOf("👩🏿", "👨🏿", "🧑🏿", "👧🏿", "🧑🏿‍🌾", "🌿", "☕", "☀️", "🚜", "🏠", "🐓", "🐐", "🐦", "🐕")

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CreateProfileScreen(
    onProfileCreated: () -> Unit,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    var name by remember { mutableStateOf("") }
    var selectedEmoji by remember { mutableStateOf(Emojis.first()) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.create_profile), fontWeight = FontWeight.Bold) },
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
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            OutlinedTextField(
                value = name,
                onValueChange = { name = it },
                label = { Text(stringResource(R.string.profile_name)) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                shape = RoundedCornerShape(12.dp)
            )

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = stringResource(R.string.choose_emoji),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Medium,
                modifier = Modifier.align(Alignment.Start)
            )

            Spacer(modifier = Modifier.height(16.dp))

            LazyVerticalGrid(
                columns = GridCells.Adaptive(minSize = 64.dp),
                contentPadding = PaddingValues(8.dp),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier.weight(1f)
            ) {
                items(Emojis) { emoji ->
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier
                            .size(64.dp)
                            .clip(CircleShape)
                            .background(
                                if (selectedEmoji == emoji) MaterialTheme.colorScheme.primaryContainer
                                else MaterialTheme.colorScheme.surfaceVariant
                            )
                            .clickable { selectedEmoji = emoji }
                    ) {
                        Text(text = emoji, fontSize = 32.sp)
                    }
                }
            }

            Button(
                onClick = {
                    if (name.isNotBlank()) {
                        coroutineScope.launch {
                            val db = AppDatabase.getDatabase(context)
                            db.profileDao().insert(Profile(name = name, emoji = selectedEmoji))
                            onProfileCreated()
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(64.dp),
                enabled = name.isNotBlank(),
                shape = RoundedCornerShape(16.dp)
            ) {
                Text(stringResource(R.string.create_profile), fontSize = 18.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}