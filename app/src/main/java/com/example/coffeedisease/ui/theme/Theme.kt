package com.example.coffeedisease.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val LightColorScheme = lightColorScheme(
    primary = PrimaryGreen,
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = LightGreen,
    onPrimaryContainer = DarkGreen,

    secondary = CoffeeBrown,
    onSecondary = Color(0xFFFFFFFF),

    background = MainBackground,
    onBackground = PrimaryText,

    surface = CardBackground,
    onSurface = PrimaryText,
    surfaceVariant = MainBackground,
    onSurfaceVariant = SecondaryText,

    outline = Borders,
    error = StatusDisease,
    onError = Color(0xFFFFFFFF)
)

// We force light theme for simplicity in outdoor visibility, or map dark theme to the same colors to keep the branding consistent.
@Composable
fun CoffeeDiseaseDetectionTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = LightColorScheme

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = false
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
