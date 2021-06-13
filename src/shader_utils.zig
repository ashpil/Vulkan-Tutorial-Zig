const c = @cImport({
    @cInclude("shaderc/shaderc.h");
});

const std = @import("std");
const print = std.debug.print;

pub const CompilationError = error {
    invalid_stage,
    compilation_error,
    internal_error,
    null_result_object,
    invalid_assembly,
    validation_error,
    transformation_error,
    configuration_error,
};

const Error = std.mem.Allocator.Error || CompilationError;

pub fn glslStringToSPV(allocator: *std.mem.Allocator, glsl: []const u8) Error![]const u8 {
    const compiler = c.shaderc_compiler_initialize();
    defer c.shaderc_compiler_release(compiler);

    const options = c.shaderc_compile_options_initialize();
    defer c.shaderc_compile_options_release(options);

    const result = c.shaderc_compile_into_spv(compiler, glsl.ptr, glsl.len, c.shaderc_shader_kind.shaderc_glsl_infer_from_source, "a.glsl", "main", options);
    defer c.shaderc_result_release(result);

    const status = c.shaderc_result_get_compilation_status(result);
    if (status != c.shaderc_compilation_status._success) {
        const message = c.shaderc_result_get_error_message(result);
        std.log.warn("{s}", .{ message });
        return status_to_err(status);
    }
    const len = c.shaderc_result_get_length(result);
    const bytes = c.shaderc_result_get_bytes(result);
    const copied = try allocator.alloc(u8, len);
    std.mem.copy(u8, copied[0..len], bytes[0..len]);

    return copied;
}

pub fn fileToSPV(allocator: *std.mem.Allocator, comptime path: []const u8) Error![]const u8 {
    return glslStringToSPV(allocator, @embedFile(path));
}

fn status_to_err(status: c.shaderc_compilation_status) CompilationError {
    switch (status) {
        c.shaderc_compilation_status._invalid_stage => return CompilationError.invalid_stage,
        c.shaderc_compilation_status._compilation_error => return CompilationError.compilation_error,
        c.shaderc_compilation_status._internal_error => return CompilationError.internal_error,
        c.shaderc_compilation_status._null_result_object => return CompilationError.null_result_object,
        c.shaderc_compilation_status._invalid_assembly => return CompilationError.invalid_assembly,
        c.shaderc_compilation_status._validation_error => return CompilationError.validation_error,
        c.shaderc_compilation_status._transformation_error => return CompilationError.transformation_error,
        c.shaderc_compilation_status._configuration_error => return CompilationError.configuration_error,
        else => unreachable,
    }
}
