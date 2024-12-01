#define GLFW_DLL
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ProgramManagers/ParseProgramParameters.cuh"
#include "ProgramManagers/CreateManagerInstance.cuh"

#include <cstdio>
#include <stdexcept>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <thrust/host_vector.h>
#include <vector>
#include <iostream>
#include <cstdlib>

#define NUMBER_OF_COLORS 20

// Membership to color mapping
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

bool mousePressed = false;
double lastX = 0, lastY = 0;
glm::mat4 rotationMatrix = glm::mat4(1.0f);

// Raw shader source as strings
const char* vertexShaderSource = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;    // Point position
layout(location = 1) in vec3 aColor;  // Point color
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vertexColor;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    vertexColor = aColor;
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vertexColor, 1.0);
}
)glsl";

GLuint CompileShader(GLenum type, const char* source) {
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, NULL);
	glCompileShader(shader);

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

void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mousePressed = true;
			glfwGetCursorPos(window, &lastX, &lastY);
		}
		else if (action == GLFW_RELEASE) {
			mousePressed = false;
		}
	}
}

void CursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (!mousePressed) return;

	double dx = xpos - lastX;
	double dy = ypos - lastY;

	lastX = xpos;
	lastY = ypos;

	float angleX = glm::radians((float)dx * 0.5f);
	float angleY = glm::radians((float)dy * 0.5f);

	glm::mat4 rotX = glm::rotate(glm::mat4(1.0f), angleX, glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 rotY = glm::rotate(glm::mat4(1.0f), angleY, glm::vec3(1.0f, 0.0f, 0.0f));

	rotationMatrix = rotX * rotY * rotationMatrix;
}

int Visualize(thrust::host_vector<glm::vec3> glmPoints, thrust::host_vector<glm::vec3> glmCentroids, thrust::host_vector<size_t>& membership) {
	if (!glfwInit()) return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(800, 600, "K Means Clustering Visualization", NULL, NULL);

	if (!window) {
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	if (glewInit() != GLEW_OK) return -1;

	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetCursorPosCallback(window, CursorPositionCallback);

	GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, vertexShaderSource);
	GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	int success;
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		exit(-1);
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	thrust::host_vector<glm::vec3> colors(glmPoints.size());

	for (size_t i = 0; i < glmPoints.size(); ++i) {
		colors[i] = membershipColors[membership[i] % NUMBER_OF_COLORS];
	}

	GLuint VBOs[2], VAO;
	glGenBuffers(2, VBOs);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
	glBufferData(GL_ARRAY_BUFFER, glmPoints.size() * sizeof(glm::vec3), glmPoints.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
	glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec3), colors.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);

	glEnable(GL_PROGRAM_POINT_SIZE);
	glPointSize(5.0f);

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glm::mat4 model = rotationMatrix;
		glm::mat4 view = glm::lookAt(glm::vec3(15.0f, 15.0f, 15.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

		glUseProgram(shaderProgram);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

		glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS, 0, glmPoints.size());

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(2, VBOs);
	glfwTerminate();
	return 0;
}

int main(int argc, char* argv[])
{
	auto parameters = ParseProgramParameters(argc, argv);

	if (!parameters.Success) {
		return 1;
	}

	std::unique_ptr<IProgramManager> manager{};

	try
	{
		manager = CreateManagerInstance(parameters);
		auto membership = manager->StartComputation();
		manager->SaveDataToOutputFile(membership);
		manager->DisplaySummary();

		if (manager->GetDimension() == 3) {
			auto glmPoints = manager->GetGlmPoints();
			auto glmCentroids = manager->GetGlmCentroids();

			Visualize(glmPoints, glmCentroids, membership);
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what();
		return 1;
	}
}