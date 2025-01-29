/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#pragma once

namespace IC {

template <class T> class Director {
private:
  InstructionCompiler<T>* m_generator;

public:
  // Ctor
  explicit Director(InstructionCompiler<T>* ig) : m_generator(ig) {}
  // Instruction write to file(s)
  void writeTxt(std::string& fname) {}
  void writeHpp(std::string& fname) {}
  void writeBin(std::string& fname) {}
  // Instruction generate
  void generate(void) {
    // static_cast<T*>(m_generator)->generate();
    m_generator->generate();
  }
};

}; // namespace IC