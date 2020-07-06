const path = require('path')
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = {
  mode: 'development',

  plugins: [
    new MiniCssExtractPlugin({
      filename: '../css/[name].css',
    }),
  ],

  entry: "./src/index.js",

  output: {
    //publicPath: 'static',
    path: path.join(__dirname, 'static/dist'),
    filename: '[name].bundle.js'
  },

  module: {
    rules: [
      {
        test: /\.scss$/,
        use: [
          {
            loader: MiniCssExtractPlugin.loader,
            options: { publicPath: "static" },
          },
          { loader: "css-loader", options: {url:false} },
          "sass-loader"
        ],
      },
    ],
  },

  node: {
    fs: 'empty',
    net: 'empty',
    tls: 'empty',
  },
};
