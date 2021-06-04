const std = @import("std");
const shader_utils = @import("shader_utils.zig");

const zalgebra = @import("zalgebra");
const Vec2 = zalgebra.vec2;
const Vec3 = zalgebra.vec3;

const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", {});
    @cInclude("GLFW/glfw3.h");
});

const VulkanError = error {
    InstanceCreateError,
    UnavailableValidationLayers,
    NoVulkanDevices,
    NoSuitableDevices,
    LogicalDeviceCreateFail,
    WindowSurfaceCreateFail,
    SwapChainCreateFail,
    ImageViewCreateFail,
    ShaderModuleCreateFail,
    PipelineLayoutCreateFail,
    RenderPassCreateFail,
    GraphicsPipelineCreateFail,
    FramebufferCreateFail,
    CommandPoolCreateError,
    CommandBufferAllocateError,
    CommandBufferRecordFail,
    CommandBufferRecordError,
    SyncObjectsCreateError,
    DrawCommandSubmitFail,
    SwapChainImageAcquireFail,
    SwapChainImagePresentFail,
    VertexBufferCreateFail,
    MemoryTypeFindFail,
    VertexBufferMemoryAllocateFail,
};

const AllocError = std.mem.Allocator.Error;

const WIDTH = 800;
const HEIGHT = 600;

const MAX_FRAMES_IN_FLIGHT = 2;

const validationLayers = [_][]const u8{ "VK_LAYER_KHRONOS_validation" };
const ptrLayers = comptime comp: {
    var layers = [validationLayers.len][*]const u8 { undefined };
    for (validationLayers) |layer, i| {
        layers[i] = layer.ptr;
    }
    break :comp layers;
};
const deviceExtensions = [_][]const u8{ c.VK_KHR_SWAPCHAIN_EXTENSION_NAME };
const ptrExtensions = comptime comp: {
    var extensions = [deviceExtensions.len][*]const u8 { undefined };
    for (deviceExtensions) |extension, i| {
        extensions[i] = extension.ptr;
    }
    break :comp extensions;
};
const enableValidationLayers = @import("builtin").mode == std.builtin.Mode.Debug;

const Vertex = extern struct {
    pos: Vec2,
    color: Vec3,

    fn getBindingDescription() c.VkVertexInputBindingDescription {
        const bindingDescription = c.VkVertexInputBindingDescription {
            .binding = 0,
            .stride = @sizeOf(Vertex),
            .inputRate = c.VkVertexInputRate.VK_VERTEX_INPUT_RATE_VERTEX,
        };

        return bindingDescription;
    }

    fn getAttributeDescriptions() [2]c.VkVertexInputAttributeDescription {
        const attributeDescriptionPos = c.VkVertexInputAttributeDescription {
            .binding = 0,
            .location = 0,
            .format = c.VkFormat.VK_FORMAT_R32G32_SFLOAT,
            .offset = @bitOffsetOf(Vertex, "pos"),
        };

        const attributeDescriptionColor = c.VkVertexInputAttributeDescription {
            .binding = 0,
            .location = 1,
            .format = c.VkFormat.VK_FORMAT_R32G32B32_SFLOAT,
            .offset = @bitOffsetOf(Vertex, "color"),
        };

        return .{ attributeDescriptionPos, attributeDescriptionColor };
    }
};

const vertices = [_]Vertex {
    Vertex { .pos = Vec2.new(0.0, -0.5), .color = Vec3.new(1.0, 0.0, 0.0) },
    Vertex { .pos = Vec2.new(0.5, 0.5), .color = Vec3.new(0.0, 1.0, 0.0) },
    Vertex { .pos = Vec2.new(-0.5, 0.5), .color = Vec3.new(0.0, 0.0, 1.0) },
};

const SwapChainSupportDetails = struct {
    capabilities: c.VkSurfaceCapabilitiesKHR,
    formats: []c.VkSurfaceFormatKHR,
    presentModes: []c.VkPresentModeKHR,

    fn querySwapChainSupport(allocator: *std.mem.Allocator, device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR) AllocError!SwapChainSupportDetails {
        var capabilities: c.VkSurfaceCapabilitiesKHR = undefined;
        _ = c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities);

        var formatCount: u32 = 0;
        _ = c.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, null);
        const formats = try allocator.alloc(c.VkSurfaceFormatKHR, formatCount);
        if (formatCount != 0) {
            _ = c.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, formats.ptr);
        }

        var presentModeCount: u32 = 0;
        _ = c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, null);
        const presentModes = try allocator.alloc(c.VkPresentModeKHR, presentModeCount);
        if (presentModeCount != 0) {
            _ = c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, presentModes.ptr);
        }

        return SwapChainSupportDetails {
            .capabilities = capabilities,
            .formats = formats,
            .presentModes = presentModes,
        };
    }

    fn deinit(self: *const SwapChainSupportDetails, allocator: *std.mem.Allocator) void {
        allocator.free(self.formats);
        allocator.free(self.presentModes);
    }
};

const QueueFamilyIndices = struct {
    graphicsFamily: ?u32 = null,
    presentFamily: ?u32 = null,

    fn findQueueFamilies(allocator: *std.mem.Allocator, device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR) AllocError!QueueFamilyIndices {
        var indices = QueueFamilyIndices {};
        var queueFamilyCount: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, null);

        const queueFamilies = try allocator.alloc(c.VkQueueFamilyProperties, queueFamilyCount);
        defer allocator.free(queueFamilies);
        c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.ptr);
        for (queueFamilies) |queueFamily, i| {
            if (queueFamily.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0) {
                indices.graphicsFamily = @intCast(u32, i);
            }
            var presentSupport = c.VK_FALSE;
            _ = c.vkGetPhysicalDeviceSurfaceSupportKHR(device, @intCast(u32, i), surface, &presentSupport);
            if (presentSupport == c.VK_TRUE) indices.presentFamily = @intCast(u32, i);
            if (indices.isComplete()) break;
        }
        return indices;
    }

    fn isComplete(self: *const QueueFamilyIndices) bool {
        return self.graphicsFamily != null and self.presentFamily != null;
    }
};

const HelloTriangleApplication = struct {
    allocator: *std.mem.Allocator,

    window: *c.GLFWwindow = undefined,
    instance: c.VkInstance = undefined,
    physicalDevice: c.VkPhysicalDevice = undefined,
    device: c.VkDevice = undefined,
    surface: c.VkSurfaceKHR = undefined,

    graphicsQueue: c.VkQueue = undefined,
    presentQueue: c.VkQueue = undefined,

    swapChain: c.VkSwapchainKHR = undefined,
    swapChainImages: []c.VkImage = undefined,
    swapChainImageViews: []c.VkImageView = undefined,
    swapChainImageFormat: c.VkFormat = undefined,
    swapChainExtent: c.VkExtent2D = undefined,
    swapChainFramebuffers: []c.VkFramebuffer = undefined,

    renderPass: c.VkRenderPass = undefined,
    pipelineLayout: c.VkPipelineLayout = undefined,
    graphicsPipeline: c.VkPipeline = undefined,

    commandPool: c.VkCommandPool = undefined,
    commandBuffers: []c.VkCommandBuffer = undefined,

    imageAvailableSemaphores: [MAX_FRAMES_IN_FLIGHT]c.VkSemaphore = undefined,
    renderFinishedSemaphores: [MAX_FRAMES_IN_FLIGHT]c.VkSemaphore = undefined,
    inFlightFences: [MAX_FRAMES_IN_FLIGHT]c.VkFence = undefined,
    imagesInFlight: []?c.VkFence = undefined,
    currentFrame: usize = 0,

    framebufferResized: bool = false,

    vertexBuffer: c.VkBuffer = undefined,
    vertexBufferMemory: c.VkDeviceMemory = undefined,

    pub fn run(self: *HelloTriangleApplication) !void {
        self.initWindow();
        try self.initVulkan();
        errdefer {
            c.glfwDestroyWindow(self.window);
            c.glfwTerminate();
        }
        try self.mainLoop();
        self.cleanup();
    }

    fn initWindow(self: *HelloTriangleApplication) void {
        _ = c.glfwInit();

        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_FALSE);

        self.window = c.glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", null, null).?;
        c.glfwSetWindowUserPointer(self.window, self);
        _ = c.glfwSetFramebufferSizeCallback(self.window, framebufferResizeCallback);
    }

    fn framebufferResizeCallback(window: ?*c.GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
        const app = @ptrCast(*HelloTriangleApplication, @alignCast(8, c.glfwGetWindowUserPointer(window).?));
        app.framebufferResized = true;
    }

    fn initVulkan(self: *HelloTriangleApplication) !void {
        try self.createInstance();
        try self.createSurface();
        try self.pickPhysicalDevice();
        try self.createLogicalDevice();
        try self.createSwapChain();
        try self.createImageViews();
        try self.createRenderPass();
        try self.createGraphicsPipeline();
        try self.createFramebuffers();
        try self.createCommandPool();
        try self.createVertexBuffer();
        try self.createCommandBuffers();
        try self.createSyncObjects();
    }

    fn createVertexBuffer(self: *HelloTriangleApplication) VulkanError!void {
        const bufferInfo = c.VkBufferCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = @sizeOf(Vertex) * vertices.len,
            .usage = c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .sharingMode = c.VkSharingMode.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
            .flags = 0,
            .pNext = null,
        };
        if (c.vkCreateBuffer(self.device, &bufferInfo, null, &self.vertexBuffer) != c.VkResult.VK_SUCCESS) return VulkanError.VertexBufferCreateFail;

        var memRequirements: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(self.device, self.vertexBuffer, &memRequirements);

        const allocInfo = c.VkMemoryAllocateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = try self.findMemoryType(memRequirements.memoryTypeBits, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
            .pNext = null,
        };
        if (c.vkAllocateMemory(self.device, &allocInfo, null, &self.vertexBufferMemory) != c.VkResult.VK_SUCCESS) return VulkanError.VertexBufferMemoryAllocateFail;
        _ = c.vkBindBufferMemory(self.device, self.vertexBuffer, self.vertexBufferMemory, 0);

        var data: []Vertex = undefined;
        data.len = vertices.len;
        _ = c.vkMapMemory(self.device, self.vertexBufferMemory, 0, bufferInfo.size, 0, @ptrCast(*?*c_void, &data.ptr));
        std.mem.copy(Vertex, data, &vertices);
        c.vkUnmapMemory(self.device, self.vertexBufferMemory);
    }

    fn findMemoryType(self: *HelloTriangleApplication, typeFilter: u32, properties: c.VkMemoryPropertyFlags) VulkanError!u32 {
        var memProperties: c.VkPhysicalDeviceMemoryProperties = undefined;
        c.vkGetPhysicalDeviceMemoryProperties(self.physicalDevice, &memProperties);

        var i: u32 = 0;
        while (i < memProperties.memoryTypeCount) : (i += 1) {
            if (typeFilter & (@as(u32, 1) << @intCast(u5, i)) != 0 and (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        return VulkanError.MemoryTypeFindFail;
    }

    fn recreateSwapChain(self: *HelloTriangleApplication) (VulkanError || AllocError || shader_utils.CompilationError)!void {
        var width: c_int = 0;
        var height: c_int = 0;
        c.glfwGetFramebufferSize(self.window, &width, &height);
        while (width == 0 or height == 0) {
            c.glfwGetFramebufferSize(self.window, &width, &height);
            c.glfwWaitEvents();
        }
        _ = c.vkDeviceWaitIdle(self.device);

        self.cleanupSwapChain();

        try self.createSwapChain();
        try self.createImageViews();
        try self.createRenderPass();
        try self.createGraphicsPipeline();
        try self.createFramebuffers();
        try self.createCommandBuffers();
    }

    fn createSyncObjects(self: *HelloTriangleApplication) (VulkanError || AllocError)!void {
        self.imagesInFlight = try self.allocator.alloc(?c.VkFence, self.swapChainImages.len);
        const semaphoreInfo = c.VkSemaphoreCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
        };
        const fenceInfo = c.VkFenceCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = null,
            .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
        };

        var i: u32 = 0;
        while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
            if (c.vkCreateSemaphore(self.device, &semaphoreInfo, null, &self.imageAvailableSemaphores[i]) != c.VkResult.VK_SUCCESS or
                c.vkCreateSemaphore(self.device, &semaphoreInfo, null, &self.renderFinishedSemaphores[i]) != c.VkResult.VK_SUCCESS or
                c.vkCreateFence(self.device, &fenceInfo, null, &self.inFlightFences[i]) != c.VkResult.VK_SUCCESS) return VulkanError.SyncObjectsCreateError;
        }
    }

    fn createCommandBuffers(self: *HelloTriangleApplication) (VulkanError || AllocError)!void {
        self.commandBuffers = try self.allocator.alloc(c.VkCommandBuffer, self.swapChainFramebuffers.len);
        const allocInfo = c.VkCommandBufferAllocateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = self.commandPool,
            .level = c.VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = @intCast(u32, self.commandBuffers.len),
            .pNext = null,
        };
        if (c.vkAllocateCommandBuffers(self.device, &allocInfo, &self.commandBuffers.ptr.*) != c.VkResult.VK_SUCCESS) return VulkanError.CommandBufferAllocateError;

        for (self.commandBuffers) |commandBuffer, i| {
            const beginInfo = c.VkCommandBufferBeginInfo {
                .sType = c.VkStructureType.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = 0,
                .pInheritanceInfo = null,
                .pNext = null,
            };
            if (c.vkBeginCommandBuffer(commandBuffer, &beginInfo) != c.VkResult.VK_SUCCESS) return VulkanError.CommandBufferRecordFail;

            const renderPassInfo = c.VkRenderPassBeginInfo {
                .sType = c.VkStructureType.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = self.renderPass,
                .framebuffer = self.swapChainFramebuffers[i],
                .renderArea = c.VkRect2D {
                    .offset = c.VkOffset2D { .x = 0, .y = 0 },
                    .extent = self.swapChainExtent,
                },
                .clearValueCount = 1,
                .pClearValues = &c.VkClearValue { .color = c.VkClearColorValue { .float32 = [_]f32{ 0.0, 0.0, 0.0, 1.0 } } },
                .pNext = null,
            };
            c.vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, c.VkSubpassContents.VK_SUBPASS_CONTENTS_INLINE);
            c.vkCmdBindPipeline(commandBuffer, c.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS, self.graphicsPipeline);

            const vertexBuffers = [_]c.VkBuffer{ self.vertexBuffer };
            const offsets = [_]u64{ 0 };

            c.vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffers, &offsets);

            c.vkCmdDraw(commandBuffer, vertices.len, 1, 0, 0);

            c.vkCmdEndRenderPass(commandBuffer);
            if (c.vkEndCommandBuffer(commandBuffer) != c.VkResult.VK_SUCCESS) return VulkanError.CommandBufferRecordError;
        }
    }

    fn createCommandPool(self: *HelloTriangleApplication) (VulkanError || AllocError)!void {
        const queueFamilyIndices = try QueueFamilyIndices.findQueueFamilies(self.allocator, self.physicalDevice, self.surface);
        const poolInfo = c.VkCommandPoolCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.?,
            .flags = 0,
            .pNext = null,
        };
        if (c.vkCreateCommandPool(self.device, &poolInfo, null, &self.commandPool) != c.VkResult.VK_SUCCESS) return VulkanError.CommandPoolCreateError;
    }

    fn createFramebuffers(self: *HelloTriangleApplication) (AllocError || VulkanError)!void {
        self.swapChainFramebuffers = try self.allocator.alloc(c.VkFramebuffer, self.swapChainImageViews.len);
        for (self.swapChainImageViews) |imageView, i| {
            const framebufferInfo = c.VkFramebufferCreateInfo {
                .sType = c.VkStructureType.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = self.renderPass,
                .attachmentCount = 1,
                .pAttachments = &imageView,
                .width = self.swapChainExtent.width,
                .height = self.swapChainExtent.height,
                .layers = 1,
                .pNext = null,
                .flags = 0,
            };
            if (c.vkCreateFramebuffer(self.device, &framebufferInfo, null, &self.swapChainFramebuffers[i]) != c.VkResult.VK_SUCCESS) return VulkanError.FramebufferCreateFail;
        }
    }

    fn createRenderPass(self: *HelloTriangleApplication) VulkanError!void {
        const colorAttachment = c.VkAttachmentDescription {
            .format = self.swapChainImageFormat,
            .samples = c.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT,
            .loadOp = c.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = c.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = c.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = c.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = c.VkImageLayout.VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = c.VkImageLayout.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .flags = 0,
        };
        const colorAttachmentRef = c.VkAttachmentReference {
            .attachment = 0,
            .layout = c.VkImageLayout.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        const subpass = c.VkSubpassDescription {
            .pipelineBindPoint = c.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef,
            .pResolveAttachments = null,
            .inputAttachmentCount = 0,
            .pInputAttachments = null,
            .pDepthStencilAttachment = null,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = null,
            .flags = 0,
        };
        const dependency = c.VkSubpassDependency {
            .srcSubpass = c.VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dependencyFlags = 0,
        };
        const renderPassInfo = c.VkRenderPassCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
            .pNext = null,
            .flags = 0,
        };
        if (c.vkCreateRenderPass(self.device, &renderPassInfo, null, &self.renderPass) != c.VkResult.VK_SUCCESS) return VulkanError.RenderPassCreateFail;
    }

    fn createGraphicsPipeline(self: *HelloTriangleApplication) (shader_utils.CompilationError || VulkanError || AllocError)!void {

        const vertShaderCode = try shader_utils.fileToSPV(self.allocator, "shader.vert");
        const fragShaderCode = try shader_utils.fileToSPV(self.allocator, "shader.frag");
        defer self.allocator.free(vertShaderCode);
        defer self.allocator.free(fragShaderCode);

        const vertShaderModule = try self.createShaderModule(vertShaderCode);
        const fragShaderModule = try self.createShaderModule(fragShaderCode);
        defer c.vkDestroyShaderModule(self.device, vertShaderModule, null);
        defer c.vkDestroyShaderModule(self.device, fragShaderModule, null);

        const vertShaderStageInfo = c.VkPipelineShaderStageCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = c.VkShaderStageFlagBits.VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertShaderModule,
            .pName = "main",
            .pSpecializationInfo = null,
            .pNext = null,
            .flags = 0,
        };
        const fragShaderStageInfo = c.VkPipelineShaderStageCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = c.VkShaderStageFlagBits.VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragShaderModule,
            .pName = "main",
            .pSpecializationInfo = null,
            .pNext = null,
            .flags = 0,
        };
        const shaderStages = [_]c.VkPipelineShaderStageCreateInfo{ vertShaderStageInfo, fragShaderStageInfo };

        const bindingDescription = Vertex.getBindingDescription();
        const attributeDescriptions = Vertex.getAttributeDescriptions();
        const vertexInputInfo = c.VkPipelineVertexInputStateCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = attributeDescriptions.len,
            .pVertexAttributeDescriptions = &attributeDescriptions,
            .pNext = null,
            .flags = 0,
        };

        const inputAssembly = c.VkPipelineInputAssemblyStateCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = c.VkPrimitiveTopology.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = c.VK_FALSE,
            .pNext = null,
            .flags = 0,
        };
        const viewport = c.VkViewport {
            .x = 0.0,
            .y = 0.0,
            .width = @intToFloat(f32, self.swapChainExtent.width),
            .height = @intToFloat(f32, self.swapChainExtent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };
        const scissor = c.VkRect2D {
            .offset = c.VkOffset2D { .x = 0, .y = 0 },
            .extent = self.swapChainExtent,
        };
        const viewportState = c.VkPipelineViewportStateCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
            .pNext = null,
            .flags = 0,
        };
        const rasterizer = c.VkPipelineRasterizationStateCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = c.VK_FALSE,
            .rasterizerDiscardEnable = c.VK_FALSE,
            .polygonMode = c.VkPolygonMode.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            .cullMode = c.VK_CULL_MODE_BACK_BIT,
            .frontFace = c.VkFrontFace.VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = c.VK_FALSE,
            .depthBiasConstantFactor = 0.0,
            .depthBiasClamp = 0.0,
            .depthBiasSlopeFactor = 1.0,
            .pNext = null,
            .flags = 0,
        };
        const multisampling = c.VkPipelineMultisampleStateCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .sampleShadingEnable = c.VK_FALSE,
            .rasterizationSamples = c.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT,
            .minSampleShading = 1.0,
            .pSampleMask = null,
            .alphaToCoverageEnable = c.VK_FALSE,
            .alphaToOneEnable = c.VK_FALSE,
            .pNext = null,
            .flags = 0,
        };
        const colorBlendAttachment = c.VkPipelineColorBlendAttachmentState {
            .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
            .blendEnable = c.VK_FALSE,
            .srcColorBlendFactor = c.VkBlendFactor.VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = c.VkBlendFactor.VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = c.VkBlendOp.VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = c.VkBlendFactor.VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = c.VkBlendFactor.VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = c.VkBlendOp.VK_BLEND_OP_ADD,
        };
        const colorBlending = c.VkPipelineColorBlendStateCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = c.VK_FALSE,
            .logicOp = c.VkLogicOp.VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
            .blendConstants = [_]f32{0.0, 0.0, 0.0, 0.0},
            .pNext = null,
            .flags = 0,
        };
        const pipelineLayoutInfo = c.VkPipelineLayoutCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 0,
            .pSetLayouts = null,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = null,
            .pNext = null,
            .flags = 0,
        };
        if (c.vkCreatePipelineLayout(self.device, &pipelineLayoutInfo, null, &self.pipelineLayout) != c.VkResult.VK_SUCCESS) return VulkanError.PipelineLayoutCreateFail;
        const pipelineInfo = c.VkGraphicsPipelineCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = &shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = null,
            .pColorBlendState = &colorBlending,
            .pDynamicState = null,
            .layout = self.pipelineLayout,
            .renderPass = self.renderPass,
            .subpass = 0,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
            .pNext = null,
            .flags = 0,
            .pTessellationState = null,
        };
        if (c.vkCreateGraphicsPipelines(self.device, null, 1, &pipelineInfo, null, &self.graphicsPipeline) != c.VkResult.VK_SUCCESS) return VulkanError.GraphicsPipelineCreateFail;
    }

    fn createShaderModule(self: *HelloTriangleApplication, code: []const u8) VulkanError!c.VkShaderModule {
        const createInfo = c.VkShaderModuleCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.len,
            .pCode = @ptrCast([*]const u32, @alignCast(4, code.ptr)),
            .pNext = null,
            .flags = 0,
        };
        var shaderModule: c.VkShaderModule = undefined;
        if (c.vkCreateShaderModule(self.device, &createInfo, null, &shaderModule) != c.VkResult.VK_SUCCESS) return VulkanError.ShaderModuleCreateFail;
        return shaderModule;
    }

    fn createImageViews(self: *HelloTriangleApplication) !void {
        self.swapChainImageViews = try self.allocator.alloc(c.VkImageView, self.swapChainImages.len);
        for (self.swapChainImages) |swapChainImage, i| {
            const createInfo = c.VkImageViewCreateInfo {
                .sType = c.VkStructureType.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = swapChainImage,
                .viewType = c.VkImageViewType.VK_IMAGE_VIEW_TYPE_2D,
                .format = self.swapChainImageFormat,
                .components = c.VkComponentMapping {
                    .r = c.VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = c.VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = c.VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = c.VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                },
                .subresourceRange = c.VkImageSubresourceRange {
                    .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .pNext = null,
                .flags = 0,
            };
            if (c.vkCreateImageView(self.device, &createInfo, null, &self.swapChainImageViews[i]) != c.VkResult.VK_SUCCESS) return VulkanError.ImageViewCreateFail;
        }
    }

    fn createSwapChain(self: *HelloTriangleApplication) !void {
        const swapChainSupport = try SwapChainSupportDetails.querySwapChainSupport(self.allocator, self.physicalDevice, self.surface);
        defer swapChainSupport.deinit(self.allocator);

        const surfaceFormat = self.chooseSwapSurfaceFormat(swapChainSupport.formats);
        const presentMode = self.chooseSwapPresentMode(swapChainSupport.presentModes);
        const extent = self.chooseSwapExtent(&swapChainSupport.capabilities);
        var imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 and imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        const indices = try QueueFamilyIndices.findQueueFamilies(self.allocator, self.physicalDevice, self.surface);
        const queueFamilyIndices = [2]u32 {indices.graphicsFamily.?, indices.presentFamily.?};
        const sameQueue = indices.graphicsFamily.? == indices.presentFamily.?;
        const createInfo = c.VkSwapchainCreateInfoKHR {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = self.surface,
            .minImageCount = imageCount,
            .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = if (sameQueue) c.VkSharingMode.VK_SHARING_MODE_EXCLUSIVE else c.VkSharingMode.VK_SHARING_MODE_CONCURRENT,
            .queueFamilyIndexCount = if (sameQueue) 0 else 2,
            .pQueueFamilyIndices = if (sameQueue) &queueFamilyIndices else null,
            .preTransform = swapChainSupport.capabilities.currentTransform,
            .compositeAlpha = c.VkCompositeAlphaFlagBitsKHR.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = presentMode,
            .clipped = c.VK_TRUE,
            .oldSwapchain = null,
            .pNext = null,
            .flags = 0,
        };
        if (c.vkCreateSwapchainKHR(self.device, &createInfo, null, &self.swapChain) != c.VkResult.VK_SUCCESS) return VulkanError.SwapChainCreateFail;
        _ = c.vkGetSwapchainImagesKHR(self.device, self.swapChain, &imageCount, null);
        self.swapChainImages = try self.allocator.alloc(c.VkImage, imageCount);
        _ = c.vkGetSwapchainImagesKHR(self.device, self.swapChain, &imageCount, self.swapChainImages.ptr);
        self.swapChainImageFormat = surfaceFormat.format;
        self.swapChainExtent = extent;
    }

    fn createSurface(self: *HelloTriangleApplication) !void {
        if (c.glfwCreateWindowSurface(self.instance, self.window, null, &self.surface) != c.VkResult.VK_SUCCESS) return VulkanError.WindowSurfaceCreateFail;
    }

    fn pickPhysicalDevice(self: *HelloTriangleApplication) !void {
        var deviceCount: u32 = 0;
        _ = c.vkEnumeratePhysicalDevices(self.instance, &deviceCount, null);
        if (deviceCount == 0) return VulkanError.NoVulkanDevices;
        const devices = try self.allocator.alloc(c.VkPhysicalDevice, deviceCount);
        defer self.allocator.free(devices);
        _ = c.vkEnumeratePhysicalDevices(self.instance, &deviceCount, devices.ptr);
        self.physicalDevice = for (devices) |device| {
            if (try self.isDeviceSuitable(device)) {
                break device;
            }
        } else return VulkanError.NoSuitableDevices;
    }

    fn createLogicalDevice(self: *HelloTriangleApplication) !void {
        const indices = try QueueFamilyIndices.findQueueFamilies(self.allocator, self.physicalDevice, self.surface);
        const queuePriority: f32 = 1.0;
        const queueCreateInfo = c.VkDeviceQueueCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = indices.graphicsFamily.?,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
            .pNext = null,
            .flags = 0,
        };
        const deviceFeatures = c.VkPhysicalDeviceFeatures {
            .robustBufferAccess = 0,
            .fullDrawIndexUint32 = 0,
            .imageCubeArray = 0,
            .independentBlend = 0,
            .geometryShader = 0,
            .tessellationShader = 0,
            .sampleRateShading = 0,
            .dualSrcBlend = 0,
            .logicOp = 0,
            .multiDrawIndirect = 0,
            .drawIndirectFirstInstance = 0,
            .depthClamp = 0,
            .depthBiasClamp = 0,
            .fillModeNonSolid = 0,
            .depthBounds = 0,
            .wideLines = 0,
            .largePoints = 0,
            .alphaToOne = 0,
            .multiViewport = 0,
            .samplerAnisotropy = 0,
            .textureCompressionETC2 = 0,
            .textureCompressionASTC_LDR = 0,
            .textureCompressionBC = 0,
            .occlusionQueryPrecise = 0,
            .pipelineStatisticsQuery = 0,
            .vertexPipelineStoresAndAtomics = 0,
            .fragmentStoresAndAtomics = 0,
            .shaderTessellationAndGeometryPointSize = 0,
            .shaderImageGatherExtended = 0,
            .shaderStorageImageExtendedFormats = 0,
            .shaderStorageImageMultisample = 0,
            .shaderStorageImageReadWithoutFormat = 0,
            .shaderStorageImageWriteWithoutFormat = 0,
            .shaderUniformBufferArrayDynamicIndexing = 0,
            .shaderSampledImageArrayDynamicIndexing = 0,
            .shaderStorageBufferArrayDynamicIndexing = 0,
            .shaderStorageImageArrayDynamicIndexing = 0,
            .shaderClipDistance = 0,
            .shaderCullDistance = 0,
            .shaderFloat64 = 0,
            .shaderInt64 = 0,
            .shaderInt16 = 0,
            .shaderResourceResidency = 0,
            .shaderResourceMinLod = 0,
            .sparseBinding = 0,
            .sparseResidencyBuffer = 0,
            .sparseResidencyImage2D = 0,
            .sparseResidencyImage3D = 0,
            .sparseResidency2Samples = 0,
            .sparseResidency4Samples = 0,
            .sparseResidency8Samples = 0,
            .sparseResidency16Samples = 0,
            .sparseResidencyAliased = 0,
            .variableMultisampleRate = 0,
            .inheritedQueries = 0,
        };
        const createInfo = c.VkDeviceCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pQueueCreateInfos = &queueCreateInfo,
            .queueCreateInfoCount = 1,
            .pEnabledFeatures = &deviceFeatures,
            .enabledExtensionCount = deviceExtensions.len,
            .ppEnabledExtensionNames = &ptrExtensions,
            .enabledLayerCount = if (enableValidationLayers) validationLayers.len else 0,
            .ppEnabledLayerNames = if (enableValidationLayers) &ptrLayers else null,
            .pNext = null,
            .flags = 0,
        };
        if (c.vkCreateDevice(self.physicalDevice, &createInfo, null, &self.device) != c.VkResult.VK_SUCCESS) return VulkanError.LogicalDeviceCreateFail;
        c.vkGetDeviceQueue(self.device, indices.graphicsFamily.?, 0, &self.graphicsQueue);
        c.vkGetDeviceQueue(self.device, indices.presentFamily.?, 0, &self.presentQueue);
    }

    fn createInstance(self: *HelloTriangleApplication) !void {
        if (enableValidationLayers and !(try checkValidationLayerSupport(self.allocator))) return VulkanError.UnavailableValidationLayers;
        const appInfo = c.VkApplicationInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "Hello Triangle",
            .applicationVersion = c.VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = c.VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = c.VK_API_VERSION_1_0,
            .pNext = null,
        };
        var glfwExtensionCount: u32 = 0;
        const glfwExtensions = c.glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        const createInfo = c.VkInstanceCreateInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = glfwExtensionCount,
            .ppEnabledExtensionNames = glfwExtensions,
            .enabledLayerCount = if (enableValidationLayers) validationLayers.len else 0,
            .ppEnabledLayerNames = if (enableValidationLayers) &ptrLayers else null,
            .flags = 0,
            .pNext = null,
        };
        if (c.vkCreateInstance(&createInfo, null, &self.instance) != c.VkResult.VK_SUCCESS) return VulkanError.InstanceCreateError;
    }

    fn mainLoop(self: *HelloTriangleApplication) (VulkanError || AllocError || shader_utils.CompilationError)!void {
        while (c.glfwWindowShouldClose(self.window) == 0) {
            c.glfwPollEvents();
            try self.drawFrame();
        }
        _ = c.vkDeviceWaitIdle(self.device);
    }

    fn drawFrame(self: *HelloTriangleApplication) (VulkanError || AllocError || shader_utils.CompilationError)!void {
        _ = c.vkWaitForFences(self.device, 1, &self.inFlightFences[self.currentFrame], c.VK_TRUE, std.math.maxInt(u64));
        var imageIndex: u32 = undefined;
        var result = c.vkAcquireNextImageKHR(self.device, self.swapChain, std.math.maxInt(u64), self.imageAvailableSemaphores[self.currentFrame], null, &imageIndex);
        if (result == c.VkResult.VK_ERROR_OUT_OF_DATE_KHR) {
            try self.recreateSwapChain();
            return;
        } else if (result != c.VkResult.VK_SUCCESS and result != c.VkResult.VK_SUBOPTIMAL_KHR) {
            return VulkanError.SwapChainImageAcquireFail;
        }
        if (self.imagesInFlight[imageIndex]) |image| _ = c.vkWaitForFences(self.device, 1, &self.imagesInFlight[imageIndex].?, c.VK_TRUE, std.math.maxInt(u64));
        self.imagesInFlight[imageIndex] = self.inFlightFences[self.currentFrame];

        const signalSemaphores = [_]c.VkSemaphore { self.renderFinishedSemaphores[self.currentFrame] };
        const waitSemaphores = [_]c.VkSemaphore { self.imageAvailableSemaphores[self.currentFrame] };
        const submitInfo = c.VkSubmitInfo {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = waitSemaphores.len,
            .pWaitSemaphores = &waitSemaphores,
            .pWaitDstStageMask = &@intCast(u32, c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
            .commandBufferCount = 1,
            .pCommandBuffers = &self.commandBuffers[imageIndex],
            .signalSemaphoreCount = signalSemaphores.len,
            .pSignalSemaphores = &signalSemaphores,
            .pNext = null,
        };
        _ = c.vkResetFences(self.device, 1, &self.inFlightFences[self.currentFrame]);
        if (c.vkQueueSubmit(self.graphicsQueue, 1, &submitInfo, self.inFlightFences[self.currentFrame]) != c.VkResult.VK_SUCCESS) return VulkanError.DrawCommandSubmitFail;
        const swapChains = [_]c.VkSwapchainKHR { self.swapChain };
        const presentInfo = c.VkPresentInfoKHR {
            .sType = c.VkStructureType.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = signalSemaphores.len,
            .pWaitSemaphores = &signalSemaphores,
            .swapchainCount = swapChains.len,
            .pSwapchains = &swapChains,
            .pImageIndices = &imageIndex,
            .pResults = null,
            .pNext = null,
        };
        result = c.vkQueuePresentKHR(self.presentQueue, &presentInfo);
        if (result == c.VkResult.VK_ERROR_OUT_OF_DATE_KHR or result == c.VkResult.VK_SUBOPTIMAL_KHR) {
            self.framebufferResized = false;
            try self.recreateSwapChain();
            return;
        } else if (result != c.VkResult.VK_SUCCESS) {
            return VulkanError.SwapChainImagePresentFail;
        }

        self.currentFrame = (self.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn cleanupSwapChain(self: *HelloTriangleApplication) void {
        for (self.swapChainFramebuffers) |framebuffer| {
            c.vkDestroyFramebuffer(self.device, framebuffer, null);
        }
        self.allocator.free(self.swapChainFramebuffers);

        c.vkFreeCommandBuffers(self.device, self.commandPool, @intCast(u32, self.commandBuffers.len), self.commandBuffers.ptr);
        self.allocator.free(self.commandBuffers);

        c.vkDestroyPipeline(self.device, self.graphicsPipeline, null);
        c.vkDestroyPipelineLayout(self.device, self.pipelineLayout, null);
        c.vkDestroyRenderPass(self.device, self.renderPass, null);

        for (self.swapChainImageViews) |imageView| {
            c.vkDestroyImageView(self.device, imageView, null);
        }

        self.allocator.free(self.swapChainImages);
        self.allocator.free(self.swapChainImageViews);
        c.vkDestroySwapchainKHR(self.device, self.swapChain, null);
    }

    fn cleanup(self: *HelloTriangleApplication) void {
        self.cleanupSwapChain();

        c.vkDestroyBuffer(self.device, self.vertexBuffer, null);
        c.vkFreeMemory(self.device, self.vertexBufferMemory, null);

        var i: u32 = 0;
        while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
            c.vkDestroySemaphore(self.device, self.renderFinishedSemaphores[i], null);
            c.vkDestroySemaphore(self.device, self.imageAvailableSemaphores[i], null);
            c.vkDestroyFence(self.device, self.inFlightFences[i], null);
        }

        self.allocator.free(self.imagesInFlight);

        c.vkDestroyCommandPool(self.device, self.commandPool, null);

        c.vkDestroyDevice(self.device, null);

        c.vkDestroySurfaceKHR(self.instance, self.surface, null);
        c.vkDestroyInstance(self.instance, null);

        c.glfwDestroyWindow(self.window);

        c.glfwTerminate();
    }

    fn isDeviceSuitable(self: *HelloTriangleApplication, device: c.VkPhysicalDevice) AllocError!bool {
        const indices = try QueueFamilyIndices.findQueueFamilies(self.allocator, device, self.surface);
        var extensionsSupported = try checkDeviceExtensionSupport(self.allocator, device);
        var swapChainAdequate = false;
        if (extensionsSupported) {
            const swapChainSupport = try SwapChainSupportDetails.querySwapChainSupport(self.allocator, device, self.surface);
            defer swapChainSupport.deinit(self.allocator);
            swapChainAdequate = swapChainSupport.formats.len != 0 and swapChainSupport.presentModes.len != 0;
        }
        return indices.isComplete() and extensionsSupported and swapChainAdequate;
    }
    fn chooseSwapSurfaceFormat(self: *HelloTriangleApplication, availableFormats: []const c.VkSurfaceFormatKHR) c.VkSurfaceFormatKHR {
        for (availableFormats) |availableFormat| {
            if (availableFormat.format == c.VkFormat.VK_FORMAT_B8G8R8A8_SRGB and availableFormat.colorSpace == c.VkColorSpaceKHR.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    fn chooseSwapPresentMode(self: *HelloTriangleApplication, availablePresentModes: []const c.VkPresentModeKHR) c.VkPresentModeKHR {
        for (availablePresentModes) |availablePresentMode| {
            if (availablePresentMode == c.VkPresentModeKHR.VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }
        return c.VkPresentModeKHR.VK_PRESENT_MODE_FIFO_KHR;
    }

    fn chooseSwapExtent(self: *HelloTriangleApplication, capabilities: *const c.VkSurfaceCapabilitiesKHR) c.VkExtent2D {
        if (capabilities.currentExtent.width != std.math.maxInt(u32)) {
            return capabilities.currentExtent;
        } else {
            var width: c_int = 0;
            var height: c_int = 0;
            c.glfwGetFramebufferSize(self.window, &width, &height);
            return c.VkExtent2D {
                .width = std.math.clamp(@intCast(u32, width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                .height = std.math.clamp(@intCast(u32, height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
            };
        }
    }
};


fn checkValidationLayerSupport(allocator: *std.mem.Allocator) AllocError!bool {
    var layerCount: u32 = 0;
    _ = c.vkEnumerateInstanceLayerProperties(&layerCount, null);

    const availableLayers = try allocator.alloc(c.VkLayerProperties, layerCount);
    defer allocator.free(availableLayers);
    _ = c.vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.ptr);
    for (validationLayers) |layerName| {
        const layerFound = for (availableLayers) |layerProperties| {
            if (std.mem.eql(u8, layerName, std.mem.sliceTo(&layerProperties.layerName, 0))) {
                break true;
            }
        } else false;

        if (!layerFound) return false;
    }
    return true;
}


fn checkDeviceExtensionSupport(allocator: *std.mem.Allocator, device: c.VkPhysicalDevice) AllocError!bool {
    var extensionCount: u32 = 0;
    _ = c.vkEnumerateDeviceExtensionProperties(device, null, &extensionCount, null);

    const availableExtensions = try allocator.alloc(c.VkExtensionProperties, extensionCount);
    defer allocator.free(availableExtensions);
    _ = c.vkEnumerateDeviceExtensionProperties(device, null, &extensionCount, availableExtensions.ptr);
    for (deviceExtensions) |extensionName| {
        const extensionFound = for (availableExtensions) |extension| {
            if (std.mem.eql(u8, extensionName, std.mem.sliceTo(&extension.extensionName, 0))) {
                break true;
            }
        } else false;

        if (!extensionFound) return false;
    }
    return true;
}

pub fn main() anyerror!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var app = HelloTriangleApplication {
        .allocator = &gpa.allocator,
    };
    try app.run();
    std.log.info("Reached program end!", .{});
}
