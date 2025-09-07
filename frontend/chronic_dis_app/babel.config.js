module.exports = function (api) {
  api.cache(true);
  return {
    presets: ["babel-preset-expo"],
    plugins: [
      // This line is required for react-native-reanimated
      "react-native-reanimated/plugin",
    ],
  };
};