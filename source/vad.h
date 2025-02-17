#ifndef VAD_H
#define VAD_H

#include <onnxruntime_cxx_api.h>
#include "silero_vad.ort.h"
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>

class VadIterator
{
private:
    // OnnxRuntime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info;

    // Model configuration
    int sample_rate;                // Sample rate of the audio input
    int window_size_samples;        // Number of samples in each window
    float high_threshold;           // Threshold to start detecting speech
    float low_threshold;            // Threshold to stop detecting speech
    int min_speech_samples;         // Minimum number of samples required to consider it as speech
    int speech_pad_samples;         // Minimum number of samples to keep the speech state active after speech ends

    // Model states
    std::vector<float> input;       // Input buffer for the model
    std::vector<float> _state;      // Internal state of the model
    std::vector<int64_t> sr;        // Sample rate tensor for the model
    int64_t input_node_dims[2];     // Dimensions of the input tensor
    const int64_t state_node_dims[3] = { 2, 1, 128 }; // Dimensions of the state tensor
    const int64_t sr_node_dims[1] = { 1 }; // Dimensions of the sample rate tensor

    // Inputs and outputs
    std::vector<Ort::Value> ort_inputs;     // Input values for the model
    std::vector<Ort::Value> ort_outputs;    // Output values from the model
    std::vector<const char*> input_node_names = { "input", "state", "sr" }; // Names of input nodes
    std::vector<const char*> output_node_names = { "output", "stateN" }; // Names of output nodes

    // Internal state
    float speech_prob;            // Probability of speech from the model
    bool triggered;               // Flag indicating if speech is currently detected
    unsigned int current_sample;  // Current sample index
    unsigned int speech_start;    // Sample index where speech started
    unsigned int speech_end;      // Sample index where speech ended

    // Debugging flag
    bool debug_mode;              // Flag to enable or disable debug printing

    // Initialize the number of threads for ONNX Runtime
    void init_engine_threads(int inter_threads, int intra_threads)
    {
        session_options.SetIntraOpNumThreads(inter_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    };

    // Initialize the ONNX model
    void init_onnx_model()
    {
        try {
            init_engine_threads(1, 1);
            session = std::make_unique<Ort::Session>(env, (void*)silero_vad_ort_start, silero_vad_ort_size, session_options);
        }
        catch (const Ort::Exception& e) {
            std::cerr << "Error initializing ONNX model: " << e.what() << std::endl;
            throw;
        }
    };

    // Reset the internal states of the model
    void reset_states()
    {
        std::memset(_state.data(), 0.0f, _state.size() * sizeof(float));
        triggered = false;
        current_sample = 0;
        speech_start = 0;
        speech_end = 0;
    };

    // Predict whether the current input buffer contains speech
    void predict(const std::vector<float>& data)
    {
        // Assign the input data to the input vector
        input.assign(data.begin(), data.end());

        // Create ORT tensors for input, state, and sample rate
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
        Ort::Value state_ort = Ort::Value::CreateTensor<float>(
            memory_info, _state.data(), _state.size(), state_node_dims, 3);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);

        // Clear and add inputs to the input vector
        ort_inputs.clear();
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(state_ort));
        ort_inputs.emplace_back(std::move(sr_ort));

        // Run the model and get the outputs
        ort_outputs = session->Run(
            Ort::RunOptions{ nullptr },
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

        // Get the speech probability from the output
        speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
        // Update the internal state with the new state from the output
        float* stateN = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(_state.data(), stateN, _state.size() * sizeof(float));
        
    };

    bool is_Speech()
    {
        // Increment the current sample index
        current_sample += window_size_samples;

        // Debugging: Print the speech probability
        if (debug_mode) {
            std::cout << "Current Sample: " << current_sample
                << ", Speech Probability: " << speech_prob
                << ", Triggered: " << (triggered ? "true" : "false")
                << std::endl;
        }

        // Speech Detection with Hysteresis
        if (speech_prob >= high_threshold) {
            if (!triggered) {
                speech_start = current_sample - window_size_samples; // Record the start of speech
            }
            triggered = true; // Mark speech as detected
            speech_end = 0; // Reset speech_end when speech is detected
        }
        else if (speech_prob < low_threshold) {
            // Silence Detection
            if (triggered) {
                if (speech_end == 0) {
                    speech_end = current_sample; // Record the end of speech
                }
                // Check if the duration since speech_end is greater than or equal to speech_pad_samples
                if (current_sample - speech_end >= speech_pad_samples) {
                    triggered = false; // Stop triggering speech
                    speech_end = 0; // Reset speech_end
                }
            }
        }

        // Return true if speech is detected and the duration since speech_start is greater than or equal to min_speech_samples
        if (triggered) {
            return current_sample - speech_start >= min_speech_samples;
        }

        return false; // Return false if no speech is detected
    }

public:
    // Constructor to initialize the VADIterator with various parameters
    VadIterator(
        int Sample_rate = 16000,
        int window_size_ms = 32,
        float High_threshold = 0.3,
        float Low_threshold = 0.2,
        int Min_speech_duration_ms = 0,         // minimum duration of speech (in milliseconds) required to consider it as valid speech.
        int Speech_pad_duration_ms = 500,       // duration after speech end in which speech is still considered active.
        bool Debug_mode = false
    )
        : sample_rate(Sample_rate),
        high_threshold(High_threshold),
        low_threshold(Low_threshold),
        min_speech_samples(Min_speech_duration_ms* (Sample_rate / 1000)),
        speech_pad_samples(Speech_pad_duration_ms* (Sample_rate / 1000)),
        env(ORT_LOGGING_LEVEL_WARNING, "test"), // Initialize Ort::Env
        allocator(), // Initialize Ort::AllocatorWithDefaultOptions
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU)), // Initialize Ort::MemoryInfo
        debug_mode(Debug_mode) // Initialize debug mode
    {
        window_size_samples = window_size_ms * (sample_rate / 1000); // Calculate window size in samples
        input.resize(window_size_samples); // Resize input buffer
        _state.resize(2 * 1 * 128); // Resize internal state
        sr.resize(1); // Resize sample rate vector
        sr[0] = sample_rate; // Set sample rate
        input_node_dims[0] = 1; // Set input tensor dimensions
        input_node_dims[1] = window_size_samples; // Set input tensor dimensions

        init_onnx_model(); // Initialize the ONNX model
        reset_states(); // Reset internal states
    };

    // Process the input buffer and determine if speech is detected
    bool process(const std::vector<float>& input_buffer)
    {
        if (input_buffer.size() != static_cast<size_t>(window_size_samples)) {
            std::cerr << "Input buffer size must match window size: " << window_size_samples << std::endl;
            return false; // Return false if input buffer size is incorrect
        }
        predict(input_buffer); // Call the predict method
        return is_Speech(); 
    };

    // Get the current speech probability
    float get_speech_prob() const
    {
        return speech_prob;
    };

    // Toggle debug mode
    void set_debug_mode(bool enable)
    {
        debug_mode = enable;
    };

    // Set the high threshold
    void set_high_threshold(float value)
    {
        if (value >= 0.0f && value <= 1.0f) {
            high_threshold = value;
        }
        else {
            std::cerr << "High threshold must be between 0.0 and 1.0" << std::endl;
        }
    };

    // Set the low threshold
    void set_low_threshold(float value)
    {
        if (value >= 0.0f && value <= 1.0f) {
            low_threshold = value;
        }
        else {
            std::cerr << "Low threshold must be between 0.0 and 1.0" << std::endl;
        }
    };

    // Get the high threshold
    float get_high_threshold() const
    {
        return high_threshold;
    };

    // Get the low threshold
    float get_low_threshold() const
    {
        return low_threshold;
    };

    // Get the window size in samples
    int get_window_size_samples() const
    {
        return window_size_samples;
    };

    // Get the sample rate
    int get_sample_rate() const
    {
        return sample_rate;
    };
};

#endif