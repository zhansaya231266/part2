// Controls (same idea as Part1):
//  A/D   : camera theta -/+
//  W/S   : camera height +/-
//  Q/E   : camera radius -/+
//  P     : toggle projection (perspective / orthographic)
//  R     : reset camera
//  G     : Gouraud shading (per-vertex)
//  F     : Phong shading (per-fragment)
//  M     : cycle material
//  I/K   : lightA angle -/+ (object-space)
//  J/L   : lightA radius -/+
//  U/O   : lightA height -/+
//  ESC   : exit

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <filesystem>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

struct Vec3 { float x, y, z; Vec3() :x(0), y(0), z(0) {} Vec3(float a, float b, float c) :x(a), y(b), z(c) {} };
static inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline Vec3 operator*(const Vec3& a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
static inline Vec3 cross(const Vec3& a, const Vec3& b) { return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static inline float len(const Vec3& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
static inline Vec3 normalize(const Vec3& v) { float L = len(v); if (L == 0) return Vec3(0, 0, 0); return Vec3(v.x / L, v.y / L, v.z / L); }

// globals 
std::vector<Vec3> vertices;
std::vector<std::array<int, 3>> faces;
std::vector<Vec3> faceNormals;
std::vector<Vec3> vertexNormals;

std::vector<float> positionsN; 
std::vector<float> normalsN;
std::vector<unsigned int> indices;

Vec3 modelCentroid(0, 0, 0);
float modelScale = 1.0f;

GLuint vao = 0, vboPos = 0, vboNorm = 0, ebo = 0;
size_t indexCount = 0;

int winW = 1024, winH = 768;
const float PI = 3.14159265358979323846f;

// camera 
float camTheta = 0.0f;
float camRadius = 4.0f;
float camHeight = 0.0f;
bool usePerspective = true;

// light A (object-space)
float lightA_theta = 0.0f;
float lightA_radius = 2.5f;
float lightA_height = 0.5f;

// shading/material
int shadingMode = 1; 

struct Material { float ambient[3]; float diffuse[3]; float specular[3]; float shininess; };
std::vector<Material> materials;
int currentMaterial = 0;

struct LightColor { float ambient[3]; float diffuse[3]; float specular[3]; };
LightColor globalLightColor = { {0.2f,0.2f,0.2f},{0.6f,0.6f,0.6f},{1.0f,1.0f,1.0f} };

// matrix helpers
static std::array<float, 16> createPerspectiveMatrix(float fovy_radians, float aspect, float znear, float zfar) {
    float f = 1.0f / tanf(fovy_radians * 0.5f);
    std::array<float, 16> m{};
    m[0] = f / aspect; m[4] = 0; m[8] = 0; m[12] = 0;
    m[1] = 0; m[5] = f; m[9] = 0; m[13] = 0;
    m[2] = 0; m[6] = 0; m[10] = (zfar + znear) / (znear - zfar); m[14] = (2.0f * zfar * znear) / (znear - zfar);
    m[3] = 0; m[7] = 0; m[11] = -1; m[15] = 0;
    return m;
}
static std::array<float, 16> createLookAtMatrix(float eyeX, float eyeY, float eyeZ, float centerX, float centerY, float centerZ, float upX, float upY, float upZ) {
    float fx = centerX - eyeX, fy = centerY - eyeY, fz = centerZ - eyeZ;
    float flen = sqrtf(fx * fx + fy * fy + fz * fz); if (flen == 0) flen = 1; fx /= flen; fy /= flen; fz /= flen;
    float ux = upX, uy = upY, uz = upZ; float ulen = sqrtf(ux * ux + uy * uy + uz * uz); if (ulen == 0) ulen = 1; ux /= ulen; uy /= ulen; uz /= ulen;
    float sx = fy * uz - fz * uy, sy = fz * ux - fx * uz, sz = fx * uy - fy * ux;
    float slen = sqrtf(sx * sx + sy * sy + sz * sz); if (slen == 0) slen = 1; sx /= slen; sy /= slen; sz /= slen;
    float ux2 = sy * fz - sz * fy, uy2 = sz * fx - sx * fz, uz2 = sx * fy - sy * fx;
    float tx = -(sx * eyeX + sy * eyeY + sz * eyeZ);
    float ty = -(ux2 * eyeX + uy2 * eyeY + uz2 * eyeZ);
    float tz = (fx * eyeX + fy * eyeY + fz * eyeZ);
    std::array<float, 16> m{};
    m[0] = sx; m[4] = ux2; m[8] = -fx; m[12] = 0;
    m[1] = sy; m[5] = uy2; m[9] = -fy; m[13] = 0;
    m[2] = sz; m[6] = uz2; m[10] = -fz; m[14] = 0;
    m[3] = tx; m[7] = ty; m[11] = tz; m[15] = 1;
    return m;
}

// SMF loader
static inline std::string trim_s(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

bool load_smf(const std::string& fname) {
    try { std::cout << "CWD: " << std::filesystem::current_path().string() << "\n"; }
    catch (...) {}
    std::filesystem::path p(fname);
    std::cout << "Try open: " << (p.is_absolute() ? p.string() : std::filesystem::absolute(p).string()) << "\n";
    if (!std::filesystem::exists(p)) std::cout << "File exists: NO\n"; else std::cout << "File exists: yes\n";

    std::ifstream in(fname);
    if (!in.is_open()) { std::cerr << "Cannot open " << fname << "\n"; return false; }
    vertices.clear(); faces.clear();
    std::string line; size_t ln = 0;
    while (std::getline(in, line)) {
        ln++; line = trim_s(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        std::istringstream iss(line); std::string tok;
        if (!(iss >> tok)) continue;
        if (tok == "v") {
            float x, y, z; if (!(iss >> x >> y >> z)) { std::cerr << "Bad v at " << ln << "\n"; continue; }
            vertices.emplace_back(x, y, z);
        }
        else if (tok == "f") {
            std::vector<int> idxs; std::string part;
            while (iss >> part) {
                size_t slash = part.find('/');
                std::string sidx = (slash == std::string::npos) ? part : part.substr(0, slash);
                try { int vi = std::stoi(sidx); idxs.push_back(vi - 1); }
                catch (...) { std::cerr << "Bad idx at " << ln << "\n"; }
            }
            if (idxs.size() < 3) { std::cerr << "Face <3 at " << ln << "\n"; continue; }
            for (size_t k = 1;k + 1 < idxs.size();++k) {
                int a = idxs[0], b = idxs[k], c = idxs[k + 1];
                if (a < 0 || b < 0 || c < 0) { std::cerr << "Idx negative at " << ln << "\n"; continue; }
                faces.push_back({ a,b,c });
            }
        }
    }
    in.close();
    std::cout << "Loaded: verts=" << vertices.size() << ", tris=" << faces.size() << "\n";
    return !vertices.empty() && !faces.empty();
}

// --------------------------------------------------
void computeFaceNormals() {
    faceNormals.clear();
    faceNormals.reserve(faces.size());

    for (auto& f : faces) {
        Vec3 v1 = vertices[f[0]];
        Vec3 v2 = vertices[f[1]];
        Vec3 v3 = vertices[f[2]];
        Vec3 e1 = v2 - v1;
        Vec3 e2 = v3 - v1;
        Vec3 n = cross(e1, e2); 
        float L = len(n);
        if (L == 0.0f) {
            faceNormals.push_back(Vec3(0, 0, 0));
        }
        else {
            faceNormals.push_back(n); 
        }
    }
}

void computeVertexNormals() {
    vertexNormals.assign(vertices.size(), Vec3(0, 0, 0));
    for (size_t i = 0; i < faces.size(); ++i) {
        auto& f = faces[i];
        Vec3 n = faceNormals[i]; 

        vertexNormals[f[0]] = vertexNormals[f[0]] + n;
        vertexNormals[f[1]] = vertexNormals[f[1]] + n;
        vertexNormals[f[2]] = vertexNormals[f[2]] + n;
    }
    
    for (size_t i = 0; i < vertexNormals.size(); ++i) {
        vertexNormals[i] = normalize(vertexNormals[i]);
    }
}


// prepare GPU buffers 
void prepareBuffers() {
    float sx = 0, sy = 0, sz = 0;
    for (auto& v : vertices) { sx += v.x; sy += v.y; sz += v.z; }
    float n = (float)vertices.size();
    modelCentroid = Vec3(sx / n, sy / n, sz / n);
    float maxd = 0;
    for (auto& v : vertices) { Vec3 d = v - modelCentroid; maxd = std::max(maxd, len(d)); }
    modelScale = (maxd > 0) ? maxd : 1.0f;

    computeFaceNormals();
    computeVertexNormals();

    positionsN.assign(vertices.size() * 3, 0.0f);
    normalsN.assign(vertices.size() * 3, 0.0f);
    for (size_t i = 0;i < vertices.size();++i) {
        Vec3 v = vertices[i]; Vec3 vn = vertexNormals[i];
        Vec3 vt = (v - modelCentroid) * (1.0f / modelScale);
        positionsN[3 * i + 0] = vt.x; positionsN[3 * i + 1] = vt.y; positionsN[3 * i + 2] = vt.z;
        normalsN[3 * i + 0] = vn.x; normalsN[3 * i + 1] = vn.y; normalsN[3 * i + 2] = vn.z;
    }
    indices.clear();
    indices.reserve(faces.size() * 3);
    for (auto& f : faces) { indices.push_back((unsigned int)f[0]); indices.push_back((unsigned int)f[1]); indices.push_back((unsigned int)f[2]); }
    indexCount = indices.size();

    // create VAO/VBO/EBO
    if (vao) { glDeleteVertexArrays(1, &vao); glDeleteBuffers(1, &vboPos); glDeleteBuffers(1, &vboNorm); glDeleteBuffers(1, &ebo); }
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vboPos);
    glBindBuffer(GL_ARRAY_BUFFER, vboPos);
    glBufferData(GL_ARRAY_BUFFER, positionsN.size() * sizeof(float), positionsN.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glGenBuffers(1, &vboNorm);
    glBindBuffer(GL_ARRAY_BUFFER, vboNorm);
    glBufferData(GL_ARRAY_BUFFER, normalsN.size() * sizeof(float), normalsN.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

// shader utils
GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char buf[8192]; glGetShaderInfoLog(s, 8191, nullptr, buf); std::cerr << "Shader error: " << buf << "\n"; return 0; }
    return s;
}
GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glBindAttribLocation(p, 0, "aPos");
    glBindAttribLocation(p, 1, "aNormal");
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char buf[8192]; glGetProgramInfoLog(p, 8191, nullptr, buf); std::cerr << "Link error: " << buf << "\n"; return 0; }
    return p;
}

// shaders (Gouraud, Phong) 
const char* gouraud_vs = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uModel; uniform mat4 uView; uniform mat4 uProj;
struct Material{ vec3 ambient; vec3 diffuse; vec3 specular; float shininess; };
uniform Material uMat;
struct Light{ vec3 position; vec3 ambient; vec3 diffuse; vec3 specular; };
uniform Light lightA; uniform Light lightB;
uniform vec3 uViewPos;
out vec3 vColor;
vec3 computeLight(Light L, vec3 pos, vec3 norm){
    vec3 ambient = L.ambient * uMat.ambient;
    vec3 n = normalize(norm);
    vec3 ldir = normalize(L.position - pos);
    float diff = max(dot(n, ldir), 0.0);
    vec3 diffuse = L.diffuse * (diff * uMat.diffuse);
    vec3 viewDir = normalize(uViewPos - pos);
    vec3 reflectDir = reflect(-ldir, n);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), uMat.shininess);
    vec3 specular = L.specular * (spec * uMat.specular);
    return ambient + diffuse + specular;
}
void main(){
    vec4 worldPos = uModel * vec4(aPos,1.0);
    vec3 pos = worldPos.xyz;
    vec3 normal = mat3(transpose(inverse(uModel))) * aNormal;
    vColor = computeLight(lightA, pos, normal) + computeLight(lightB, pos, normal);
    gl_Position = uProj * uView * worldPos;
}
)GLSL";

const char* gouraud_fs = R"GLSL(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main(){ FragColor = vec4(vColor,1.0); }
)GLSL";

const char* phong_vs = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uModel; uniform mat4 uView; uniform mat4 uProj;
out vec3 FragPos; out vec3 Normal;
void main(){
    vec4 worldPos = uModel * vec4(aPos,1.0);
    FragPos = worldPos.xyz;
    Normal = mat3(transpose(inverse(uModel))) * aNormal;
    gl_Position = uProj * uView * worldPos;
}
)GLSL";

const char* phong_fs = R"GLSL(
#version 330 core
struct Material{ vec3 ambient; vec3 diffuse; vec3 specular; float shininess; };
struct Light{ vec3 position; vec3 ambient; vec3 diffuse; vec3 specular; };
uniform Material uMat; uniform Light lightA; uniform Light lightB; uniform vec3 uViewPos;
in vec3 FragPos; in vec3 Normal;
out vec4 FragColor;
vec3 calc(Light L){
    vec3 ambient = L.ambient * uMat.ambient;
    vec3 n = normalize(Normal);
    vec3 ldir = normalize(L.position - FragPos);
    float diff = max(dot(n, ldir), 0.0);
    vec3 diffuse = L.diffuse * (diff * uMat.diffuse);
    vec3 viewDir = normalize(uViewPos - FragPos);
    vec3 reflectDir = reflect(-ldir, n);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), uMat.shininess);
    vec3 specular = L.specular * (spec * uMat.specular);
    return ambient + diffuse + specular;
}
void main(){
    vec3 color = calc(lightA) + calc(lightB);
    FragColor = vec4(color,1.0);
}
)GLSL";

GLuint progG = 0, progP = 0;

bool createPrograms() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, gouraud_vs);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, gouraud_fs);
    if (!vs || !fs) return false;
    progG = linkProgram(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);
    vs = compileShader(GL_VERTEX_SHADER, phong_vs);
    fs = compileShader(GL_FRAGMENT_SHADER, phong_fs);
    if (!vs || !fs) return false;
    progP = linkProgram(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);
    return progG && progP;
}

// materials 
void prepareMaterials() {
    materials.clear();
    Material m0; // red 
    m0.ambient[0] = 0.6f; m0.ambient[1] = 0.2f; m0.ambient[2] = 0.2f;
    m0.diffuse[0] = 0.9f; m0.diffuse[1] = 0.1f; m0.diffuse[2] = 0.1f;
    m0.specular[0] = 0.8f; m0.specular[1] = 0.8f; m0.specular[2] = 0.8f;
    m0.shininess = 40.0f;
    materials.push_back(m0);
    Material m1; // blue
    m1.ambient[0] = 0.05f; m1.ambient[1] = 0.05f; m1.ambient[2] = 0.2f;
    m1.diffuse[0] = 0.1f; m1.diffuse[1] = 0.1f; m1.diffuse[2] = 0.8f;
    m1.specular[0] = 0.3f; m1.specular[1] = 0.3f; m1.specular[2] = 0.3f;
    m1.shininess = 20.0f;
    materials.push_back(m1);
    Material m2; // green 
    m2.ambient[0] = 0.1f; m2.ambient[1] = 0.3f; m2.ambient[2] = 0.1f;
    m2.diffuse[0] = 0.2f; m2.diffuse[1] = 0.6f; m2.diffuse[2] = 0.2f;
    m2.specular[0] = 0.05f; m2.specular[1] = 0.05f; m2.specular[2] = 0.05f;
    m2.shininess = 5.0f;
    materials.push_back(m2);
}

// render
void renderProgram(GLuint prog, float camX, float camY, float camZ) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    float aspect = (winH == 0) ? 1.0f : (float)winW / (float)winH;
    std::array<float, 16> proj;
    if (usePerspective)
        proj = createPerspectiveMatrix(45.0f * PI / 180.0f, aspect, 0.5f, 1000.0f);
    else {
        float s = 1.5f;
        std::array<float, 16> ortho{};
        float l = -s * aspect, r = s * aspect, b = -s, t = s, n = -1000.0f, f = 1000.0f;
        ortho[0] = 2.0f / (r - l); ortho[4] = 0; ortho[8] = 0;  ortho[12] = -(r + l) / (r - l);
        ortho[1] = 0; ortho[5] = 2.0f / (t - b); ortho[9] = 0;  ortho[13] = -(t + b) / (t - b);
        ortho[2] = 0; ortho[6] = 0; ortho[10] = -2.0f / (f - n); ortho[14] = -(f + n) / (f - n);
        ortho[3] = 0; ortho[7] = 0; ortho[11] = 0; ortho[15] = 1;
        proj = ortho;
    }
    auto view = createLookAtMatrix(camX, camY, camZ, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    float modelMat[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };

    glUseProgram(prog);
    GLint locM = glGetUniformLocation(prog, "uModel");
    GLint locV = glGetUniformLocation(prog, "uView");
    GLint locP = glGetUniformLocation(prog, "uProj");
    if (locM >= 0) glUniformMatrix4fv(locM, 1, GL_FALSE, modelMat);
    if (locV >= 0) glUniformMatrix4fv(locV, 1, GL_FALSE, view.data());
    if (locP >= 0) glUniformMatrix4fv(locP, 1, GL_FALSE, proj.data());

    // set material
    Material& mat = materials[currentMaterial];
    GLint aLoc = glGetUniformLocation(prog, "uMat.ambient");
    GLint dLoc = glGetUniformLocation(prog, "uMat.diffuse");
    GLint sLoc = glGetUniformLocation(prog, "uMat.specular");
    GLint shLoc = glGetUniformLocation(prog, "uMat.shininess");
    if (aLoc >= 0) glUniform3fv(aLoc, 1, mat.ambient);
    if (dLoc >= 0) glUniform3fv(dLoc, 1, mat.diffuse);
    if (sLoc >= 0) glUniform3fv(sLoc, 1, mat.specular);
    if (shLoc >= 0) glUniform1f(shLoc, mat.shininess);

    // lights
    float lx = lightA_radius * cosf(lightA_theta);
    float ly = lightA_radius * sinf(lightA_theta);
    float lz = lightA_height;
    GLint lposA = glGetUniformLocation(prog, "lightA.position");
    GLint lambA = glGetUniformLocation(prog, "lightA.ambient");
    GLint ldiffA = glGetUniformLocation(prog, "lightA.diffuse");
    GLint lspecA = glGetUniformLocation(prog, "lightA.specular");
    if (lposA >= 0) glUniform3f(lposA, lx, ly, lz);
    if (lambA >= 0) glUniform3fv(lambA, 1, globalLightColor.ambient);
    if (ldiffA >= 0) glUniform3fv(ldiffA, 1, globalLightColor.diffuse);
    if (lspecA >= 0) glUniform3fv(lspecA, 1, globalLightColor.specular);

    GLint lposB = glGetUniformLocation(prog, "lightB.position");
    GLint lambB = glGetUniformLocation(prog, "lightB.ambient");
    GLint ldiffB = glGetUniformLocation(prog, "lightB.diffuse");
    GLint lspecB = glGetUniformLocation(prog, "lightB.specular");
    if (lposB >= 0) glUniform3f(lposB, camX, camY, camZ);
    if (lambB >= 0) glUniform3fv(lambB, 1, globalLightColor.ambient);
    if (ldiffB >= 0) glUniform3fv(ldiffB, 1, globalLightColor.diffuse);
    if (lspecB >= 0) glUniform3fv(lspecB, 1, globalLightColor.specular);

    GLint viewPosLoc = glGetUniformLocation(prog, "uViewPos");
    float viewPos[3] = { camX,camY,camZ };
    if (viewPosLoc >= 0) glUniform3fv(viewPosLoc, 1, viewPos);

    // draw
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, (GLsizei)indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

// input
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camTheta -= 0.03f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camTheta += 0.03f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camHeight += 0.03f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camHeight -= 0.03f;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camRadius = std::max(0.01f, camRadius - 0.03f);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camRadius += 0.03f;

    double t = glfwGetTime();
    static double lastG = 0, lastF = 0, lastM = 0, lastP = 0, lastR = 0;
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS && t - lastG > 0.25) { shadingMode = 0; lastG = t; std::cout << "Gouraud\n"; }
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS && t - lastF > 0.25) { shadingMode = 1; lastF = t; std::cout << "Phong\n"; }
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS && t - lastM > 0.25) { currentMaterial = (currentMaterial + 1) % materials.size(); lastM = t; std::cout << "Material " << currentMaterial << "\n"; }
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && t - lastP > 0.25) { usePerspective = !usePerspective; lastP = t; std::cout << (usePerspective ? "Perspective\n" : "Ortho\n"); }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && t - lastR > 0.25) { camRadius = 4.0f; camHeight = 0.0f; camTheta = 0.0f; lastR = t; }

    // light A controls
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) lightA_theta -= 0.04f;
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) lightA_theta += 0.04f;
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) lightA_radius = std::max(0.0f, lightA_radius - 0.03f);
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) lightA_radius += 0.03f;
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS) lightA_height -= 0.03f;
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) lightA_height += 0.03f;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    static bool flipNormals = false;
    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS) {
        static double lastN = 0;
        double t = glfwGetTime();
        if (t - lastN > 0.25) {
            flipNormals = !flipNormals;
            lastN = t;
            std::cout << "flipNormals = " << (flipNormals ? "ON" : "OFF") << "\n";
        }
    }

}

void framebuffer_cb(GLFWwindow*, int w, int h) { winW = w; winH = h; glViewport(0, 0, w, h); }

// main
int main(int argc, char** argv) {
    std::string fname;
    if (argc >= 2) fname = argv[1];

    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(winW, winH, "Part2 - Gouraud & Phong (noglm)", nullptr, nullptr);
    if (!window) { std::cerr << "Create window failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr << "GLAD init failed\n"; return -1; }
    glfwSetFramebufferSizeCallback(window, framebuffer_cb);


    vertices.clear(); faces.clear();
    int stacks = 36, sectors = 72; float radius = 1.0f;
    for (int i = 0;i <= stacks;i++) {
        float V = (float)i / (float)stacks;
        float phi = V * PI;
        for (int j = 0;j <= sectors;j++) {
            float U = (float)j / (float)sectors;
            float theta = U * 2.0f * PI;
            float x = cosf(theta) * sinf(phi);
            float y = sinf(theta) * sinf(phi);
            float z = cosf(phi);
            vertices.emplace_back(x * radius, y * radius, z * radius);
        }
    }
    int cols = sectors + 1;
    for (int i = 0;i < stacks;i++) {
        for (int j = 0;j < sectors;j++) {
            int v1 = i * cols + j;
            int v2 = v1 + cols;
            int v3 = v2 + 1;
            int v4 = v1 + 1;
            faces.push_back({ v1,v2,v3 });
            faces.push_back({ v1,v3,v4 });
        }
    }

    prepareMaterials();
    computeFaceNormals();
    computeVertexNormals();
    prepareBuffers();

    if (!createPrograms()) { std::cerr << "Shader compile/link failed\n"; return -1; }

    std::cout << "Verts: " << vertices.size() << ", Tris: " << faces.size() << "\n";
    std::cout << "Controls: A/D | W/S | Q/E | P | R | G | F | M | light controls: IJKLUO\n";

    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        // compute camera 
        float camX = camRadius * cosf(camTheta);
        float camY = camRadius * sinf(camTheta);
        float camZ = camHeight;
        GLuint prog = (shadingMode == 0) ? progG : progP;
        renderProgram(prog, camX, camY, camZ);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // cleanup
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vboPos) glDeleteBuffers(1, &vboPos);
    if (vboNorm) glDeleteBuffers(1, &vboNorm);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (progG) glDeleteProgram(progG);
    if (progP) glDeleteProgram(progP);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
