package com.example.coffeedisease

import android.os.Bundle
import android.net.Uri
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.coffeedisease.ui.theme.CoffeeDiseaseDetectionTheme
import com.example.coffeedisease.ui.screens.ProfileSelectionScreen
import com.example.coffeedisease.ui.screens.CreateProfileScreen
import com.example.coffeedisease.ui.screens.ProfileDashboardScreen
import com.example.coffeedisease.ui.screens.CameraScreen
import com.example.coffeedisease.ui.screens.ResultScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CoffeeDiseaseDetectionTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val navController = rememberNavController()

                    NavHost(navController = navController, startDestination = "profile_selection") {
                        composable("profile_selection") {
                            ProfileSelectionScreen(
                                onNavigateToCreate = { navController.navigate("create_profile") },
                                onProfileSelected = { profileId ->
                                    navController.navigate("profile_dashboard/$profileId")
                                }
                            )
                        }
                        composable("create_profile") {
                            CreateProfileScreen(
                                onProfileCreated = { navController.popBackStack() },
                                onBack = { navController.popBackStack() }
                            )
                        }
                        composable("profile_dashboard/{profileId}") { backStackEntry ->
                            val profileId = backStackEntry.arguments?.getString("profileId")?.toIntOrNull() ?: return@composable
                            ProfileDashboardScreen(
                                profileId = profileId,
                                onBack = { navController.popBackStack() },
                                onNewScan = { navController.navigate("camera/$profileId") }
                            )
                        }
                        composable("camera/{profileId}") { backStackEntry ->
                            val profileId = backStackEntry.arguments?.getString("profileId")?.toIntOrNull() ?: return@composable
                            CameraScreen(
                                profileId = profileId,
                                onBack = { navController.popBackStack() },
                                onImageCaptured = { uriString ->
                                    navController.navigate("result/$profileId?imageUri=${Uri.encode(uriString)}") {
                                        popUpTo("camera/$profileId") { inclusive = true }
                                    }
                                }
                            )
                        }
                        composable("result/{profileId}?imageUri={imageUri}") { backStackEntry ->
                            val profileId = backStackEntry.arguments?.getString("profileId")?.toIntOrNull() ?: return@composable
                            val imageUri = backStackEntry.arguments?.getString("imageUri") ?: ""
                            ResultScreen(
                                profileId = profileId,
                                imageUri = imageUri,
                                onDone = {
                                    navController.popBackStack(route = "profile_dashboard/$profileId", inclusive = false)
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}