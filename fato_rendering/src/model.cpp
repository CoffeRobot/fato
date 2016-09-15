/*****************************************************************************/
/*  Copyright (c) 2016, Alessandro Pieropan                                  */
/*  All rights reserved.                                                     */
/*                                                                           */
/*  Redistribution and use in source and binary forms, with or without       */
/*  modification, are permitted provided that the following conditions       */
/*  are met:                                                                 */
/*                                                                           */
/*  1. Redistributions of source code must retain the above copyright        */
/*  notice, this list of conditions and the following disclaimer.            */
/*                                                                           */
/*  2. Redistributions in binary form must reproduce the above copyright     */
/*  notice, this list of conditions and the following disclaimer in the      */
/*  documentation and/or other materials provided with the distribution.     */
/*                                                                           */
/*  3. Neither the name of the copyright holder nor the names of its         */
/*  contributors may be used to endorse or promote products derived from     */
/*  this software without specific prior written permission.                 */
/*                                                                           */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      */
/*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        */
/*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    */
/*  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     */
/*  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   */
/*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         */
/*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    */
/*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    */
/*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    */
/*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     */
/*****************************************************************************/
#include "model.h"

namespace rendering {

Model::Model(GLchar* path) { loadModel(path); }

void Model::draw(Shader shader) {
  for (GLuint i = 0; i < meshes.size(); i++) meshes[i].draw(shader);
}

void Model::loadModel(string path) {
  // Read file via ASSIMP
  Assimp::Importer importer;
  const aiScene* scene =
      importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);
  // Check for errors
  if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE ||
      !scene->mRootNode)  // if is Not Zero
  {
    cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << endl;
    return;
  }
  // Retrieve the directory path of the filepath
  directory = path.substr(0, path.find_last_of('/'));

  // Process ASSIMP's root node recursively
  processNode(scene->mRootNode, scene);
}

void Model::processNode(aiNode* node, const aiScene* scene) {
  // Process each mesh located at the current node
  for (GLuint i = 0; i < node->mNumMeshes; i++) {
    // The node object only contains indices to index the actual objects in
    // the scene.
    // The scene contains all the data, node is just to keep stuff organized
    // (like relations between nodes).
    aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
    meshes.push_back(processMesh(mesh, scene));
  }
  // After we've processed all of the meshes (if any) we then recursively
  // process each of the children nodes
  for (GLuint i = 0; i < node->mNumChildren; i++) {
    processNode(node->mChildren[i], scene);
  }
}

Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene) {
  // Data to fill
  vector<Vertex> vertices;
  vector<GLuint> indices;
  vector<Texture> textures;

  // Walk through each of the mesh's vertices
  for (GLuint i = 0; i < mesh->mNumVertices; i++) {
    Vertex vertex;
    glm::vec3 vector;  // We declare a placeholder vector since assimp uses
                       // its own vector class that doesn't directly convert
                       // to glm's vec3 class so we transfer the data to this
                       // placeholder glm::vec3 first.
    // Positions
    vector.x = mesh->mVertices[i].x;
    vector.y = mesh->mVertices[i].y;
    vector.z = mesh->mVertices[i].z;
    vertex.Position = vector;
    // Normals
    vector.x = mesh->mNormals[i].x;
    vector.y = mesh->mNormals[i].y;
    vector.z = mesh->mNormals[i].z;
    vertex.Normal = vector;
    // Texture Coordinates
    if (mesh->mTextureCoords[0])  // Does the mesh contain texture
                                  // coordinates?
    {
      glm::vec2 vec;
      // A vertex can contain up to 8 different texture coordinates. We thus
      // make the assumption that we won't
      // use models where a vertex can have multiple texture coordinates so we
      // always take the first set (0).
      vec.x = mesh->mTextureCoords[0][i].x;
      vec.y = mesh->mTextureCoords[0][i].y;
      vertex.TexCoords = vec;
    } else
      vertex.TexCoords = glm::vec2(0.0f, 0.0f);
    vertices.push_back(vertex);
  }
  // Now wak through each of the mesh's faces (a face is a mesh its triangle)
  // and retrieve the corresponding vertex indices.
  for (GLuint i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    // Retrieve all indices of the face and store them in the indices vector
    for (GLuint j = 0; j < face.mNumIndices; j++)
      indices.push_back(face.mIndices[j]);
  }
  // Process materials
  if (mesh->mMaterialIndex >= 0) {
    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
    // We assume a convention for sampler names in the shaders. Each diffuse
    // texture should be named
    // as 'texture_diffuseN' where N is a sequential number ranging from 1 to
    // MAX_SAMPLER_NUMBER.
    // Same applies to other texture as the following list summarizes:
    // Diffuse: texture_diffuseN
    // Specular: texture_specularN
    // Normal: texture_normalN

    // 1. Diffuse maps
    vector<Texture> diffuseMaps = loadMaterialTextures(
        material, aiTextureType_DIFFUSE, "texture_diffuse");
    textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
    // 2. Specular maps
    vector<Texture> specularMaps = loadMaterialTextures(
        material, aiTextureType_SPECULAR, "texture_specular");
    textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
  }

  // Return a mesh object created from the extracted mesh data
  return Mesh(vertices, indices, textures);
}

vector<Texture> Model::loadMaterialTextures(aiMaterial* mat, aiTextureType type,
                                     string typeName) {
  vector<Texture> textures;
  for (GLuint i = 0; i < mat->GetTextureCount(type); i++) {
    aiString str;
    mat->GetTexture(type, i, &str);
    // Check if texture was loaded before and if so, continue to next
    // iteration: skip loading a new texture
    GLboolean skip = false;
    for (GLuint j = 0; j < textures_loaded.size(); j++) {
      if (textures_loaded[j].path == str) {
        textures.push_back(textures_loaded[j]);
        skip = true;  // A texture with the same filepath has already been
                      // loaded, continue to next one. (optimization)
        break;
      }
    }
    if (!skip) {  // If texture hasn't been loaded already, load it
      Texture texture;
      texture.id = TextureFromFile(str.C_Str(), this->directory);
      texture.type = typeName;
      texture.path = str;
      textures.push_back(texture);
      this->textures_loaded.push_back(texture);  // Store it as texture loaded
                                                 // for entire model, to
                                                 // ensure we won't unnecesery
                                                 // load duplicate textures.
    }
  }
  return textures;
}

int Model::getMeshCount()
{
    return meshes.size();
}

Mesh& Model::getMesh(int index)
{
    return meshes.at(index);
}

GLint TextureFromFile(const char* path, string directory) {
  // Generate texture ID and load texture data
  string filename = string(path);
  filename = directory + '/' + filename;
  GLuint textureID;
  glGenTextures(1, &textureID);

  cv::Mat img = cv::imread(filename.c_str());
  unsigned char* img_data = img.data;

  // unsigned char* image = SOIL_load_image(filename.c_str(), &width, &height,
  // 0, SOIL_LOAD_RGB);
  // Assign texture to ID
  //TODO: quick hack to fix the odd texture size bug, check for a nicer solution
  glBindTexture(GL_TEXTURE_2D, textureID);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, img.cols);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_BGR,
               GL_UNSIGNED_BYTE, img_data);
  glGenerateMipmap(GL_TEXTURE_2D);

  // Parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  // SOIL_free_image_data(image);
  return textureID;
}


}  // end namespace
