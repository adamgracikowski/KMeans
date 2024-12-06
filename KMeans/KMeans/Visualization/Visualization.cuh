#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <thrust/host_vector.h>
#include <vector>

#define NUMBER_OF_COLORS 20

// Predefined set of colors for visualizing clusters
glm::vec3 membershipColors[] = {
	{1.0f, 0.0f, 0.0f},  // Red
	{0.0f, 1.0f, 0.0f},  // Green
	{0.0f, 0.0f, 1.0f},  // Blue
	{1.0f, 1.0f, 0.0f},  // Yellow
	{1.0f, 0.0f, 1.0f},  // Magenta
	{0.0f, 1.0f, 1.0f},  // Cyan
	{0.5f, 0.5f, 0.5f},  // Gray
	{1.0f, 0.5f, 0.0f},  // Orange
	{0.5f, 0.0f, 0.5f},  // Purple
	{0.0f, 0.5f, 0.5f},  // Teal
	{0.2f, 0.8f, 0.2f},  // Light Green
	{0.8f, 0.2f, 0.2f},  // Light Red
	{0.2f, 0.2f, 0.8f},  // Light Blue
	{0.8f, 0.8f, 0.2f},  // Light Yellow
	{0.8f, 0.2f, 0.8f},  // Light Magenta
	{0.2f, 0.8f, 0.8f},  // Light Cyan
	{0.9f, 0.4f, 0.7f},  // Pink
	{0.6f, 0.3f, 0.0f},  // Brown
	{0.3f, 0.6f, 0.9f},  // Sky Blue
	{0.7f, 0.9f, 0.3f}   // Lime
};

// Vertex shader: Transforms point positions and passes colors to the fragment shader
const char* vertexShaderSource = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;    // Point position
layout(location = 1) in vec3 aColor;  // Point color
uniform mat4 model;                   // Model transformation matrix
uniform mat4 view;                    // View (camera) transformation matrix
uniform mat4 projection;              // Projection matrix

out vec3 vertexColor;                 // Output color passed to fragment shader

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    vertexColor = aColor;
}
)glsl";

// Fragment shader: Sets the final color for each pixel (fragment)
const char* fragmentShaderSource = R"glsl(
#version 330 core
in vec3 vertexColor;                  // Input color from vertex shader
out vec4 FragColor;                   // Final color output

void main() {
    FragColor = vec4(vertexColor, 1.0);
}
)glsl";

// Initial window dimensions
int windowWidth = 800;
int windowHeight = 600;

// Global variables for mouse interaction and rotation
bool mousePressed = false;

double lastX = 0, lastY = 0;

glm::mat4 rotationMatrix = glm::mat4(1.0f);

// Callback for resizing the window; updates the OpenGL viewport to match new dimensions
void FramebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// Compiles a shader from source and checks for errors
GLuint CompileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    // Check for compilation errors
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        exit(-1);
    }
    return shader;
}

// Callback for mouse button events; tracks whether the left mouse button is pressed
void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            glfwGetCursorPos(window, &lastX, &lastY); // Save initial cursor position
        }
        else if (action == GLFW_RELEASE) {
            mousePressed = false;
        }
    }
}

// Callback for mouse movement events; updates the rotation matrix based on cursor movement
void CursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (!mousePressed) return;

    // Calculate cursor movement
    double dx = xpos - lastX;
    double dy = ypos - lastY;

    lastX = xpos;
    lastY = ypos;

    // Update rotation angles (scaled for sensitivity)
    float angleX = glm::radians((float)dx * 0.5f);
    float angleY = glm::radians((float)dy * 0.5f);

    // Create rotation matrices for X and Y axes
    glm::mat4 rotX = glm::rotate(glm::mat4(1.0f), angleX, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 rotY = glm::rotate(glm::mat4(1.0f), angleY, glm::vec3(1.0f, 0.0f, 0.0f));

    // Combine the rotations into the global rotation matrix
    rotationMatrix = rotX * rotY * rotationMatrix;
}

// Visualizes a 3D dataset of points using OpenGL
int Visualize(
    std::vector<glm::vec3>& glmPoints, 
    thrust::host_vector<size_t>& membership) 
{
    if (!glfwInit()) return -1;

    // Set OpenGL version and profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a GLFW window
    GLFWwindow* window = glfwCreateWindow(
        windowWidth, 
        windowHeight, 
        "K Means Clustering Visualization", 
        NULL, 
        NULL
    );

    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) return -1;

    // Enable depth testing to properly render overlapping points
    glEnable(GL_DEPTH_TEST);

    // Set callback functions for user interactions
    glfwSetMouseButtonCallback(window, MouseButtonCallback);
    glfwSetCursorPosCallback(window, CursorPositionCallback);
    glfwSetFramebufferSizeCallback(window, FramebufferSizeCallback);

    // Compile shaders and link them into a shader program
    GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint shaderProgram = glCreateProgram();
    
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    int success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        exit(-1);
    }

    // Delete shaders as they are now part of the program
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Compute the bounding box of the dataset
    glm::vec3 minPoint = glm::vec3(FLT_MAX);
    glm::vec3 maxPoint = glm::vec3(-FLT_MAX);
    for (const auto& point : glmPoints) {
        minPoint = glm::min(minPoint, point);
        maxPoint = glm::max(maxPoint, point);
    }

    // Compute the center and extent of the bounding box
    glm::vec3 center = (minPoint + maxPoint) * 0.5f;
    float maxExtent = glm::length(maxPoint - minPoint);

    // Map centroid membership to colors
    std::vector<glm::vec3> colors(glmPoints.size());
    for (size_t i = 0; i < glmPoints.size(); ++i) {
        colors[i] = membershipColors[membership[i] % NUMBER_OF_COLORS];
    }

    // Create and configure vertex buffers and vertex array object
    GLuint VBOs[2], VAO;
    glGenBuffers(2, VBOs);
    glGenVertexArrays(1, &VAO);

    glBindVertexArray(VAO);

    // Load point positions into the first buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
    glBufferData(GL_ARRAY_BUFFER, glmPoints.size() * sizeof(glm::vec3), glmPoints.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // Load point colors into the second buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), colors.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // Enable OpenGL point rendering and set point size
    glEnable(GL_PROGRAM_POINT_SIZE);
    glPointSize(5.0f);

    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        // Clear color and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Compute the camera's view and projection matrices
        glm::mat4 model = rotationMatrix;
        glm::vec3 cameraPosition = center + glm::vec3(0.0f, maxExtent, maxExtent * 1.5f);
        glm::mat4 view = glm::lookAt(cameraPosition, center, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(
            glm::radians(45.0f),                            // Field of view
            static_cast<float>(windowWidth) / windowHeight, // Aspect ratio
            0.5f,                                           // Near clipping plane
            maxExtent * 2.0f                                // Far clipping plane
        );

        // Pass matrices to the shader program
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Render points
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, glmPoints.size());

        // Swap buffers and process events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up OpenGL resources
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(2, VBOs);
    glfwTerminate();

    return 0;
}