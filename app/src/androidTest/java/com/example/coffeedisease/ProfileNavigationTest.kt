package com.example.coffeedisease

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import org.junit.Rule
import org.junit.Test

class ProfileNavigationTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()

    @Test fun selectingProfileShowsPhotoButton() {
        compose.onNodeWithText(compose.activity.getString(R.string.shared_profile)).performClick()
        compose.onNodeWithText(compose.activity.getString(R.string.take_photo), substring = true)
            .assertIsDisplayed()
    }
}
